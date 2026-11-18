# imgsim

**在文件夹里找出重复和近似重复的图片——而且结果是对的。**

[![CI](https://github.com/zhouchuan315-png/imgsim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhouchuan315-png/imgsim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

`imgsim` 扫描一个图片目录，为每张图算一个 64 位感知指纹，把「同一张图的不同版本」
聚成一组——改过尺寸的、重新压缩的、裁切过的、调过亮度的、转成黑白的，都能认出来。

做这个库是因为朴素感知哈希在真实图库里有两个必然踩到的坑：

| 坑 | 不处理会怎样 |
|---|---|
| **色盲**：pHash 在灰度图上计算 | 形状相同的红苹果和绿苹果被判为同一张 |
| **指纹塌缩**：平坦图像没有低频结构 | 纯白矩形和蓝白条纹图案哈希成同一个全零值 |

`imgsim` 在标准 pHash + dHash 之上加了两道校验：色相感知的颜色否决，和低信息量守卫。
在自带基准上（1000 张图 / 20 组 × 50 个变体）**20 组全部找对，零跨组混杂**。

---

## 安装

```bash
pip install imgsim-py
```

PyPI 上的 `imgsim` 是被另一个无关且已停止维护的项目占着的，所以分发名用
`imgsim-py`，但导入名仍是 `imgsim`：

```python
import imgsim
```

可选的语义模式：

```bash
pip install imgsim-py[embed]   # 会装 PyTorch + torchvision（约 2 GB）
```

## 快速开始

### 命令行

```bash
imgsim ./photos                       # 分组，生成 ./imgsim-out/report.html
imgsim ./photos -r -t 8               # 递归子目录，阈值更严
imgsim ./photos --query cat.jpg -k 5  # 还有哪些图像 cat.jpg？
imgsim ./photos -f json               # 输出 JSON，便于管道处理
```

完整参数见 `imgsim --help`。

### Python

```python
import imgsim

index = imgsim.ImageIndex()
index.add_dir("photos/", recursive=True)      # 只解码一次，多线程并行

result = index.compare(threshold=10)          # 汉明距离，满值 64
for group in result.groups:
    print(f"{len(group)} 张相似图")
    for path, distance in zip(group.paths, group.distances):
        print(f"   {distance:>6.1f}  {path}")

imgsim.save_html(result, "report.html")
```

指纹很轻，改阈值不用重新解码：

```python
index.save("photos.imgsim")                   # 每张图约 300 字节
index = imgsim.ImageIndex.load("photos.imgsim")
result = index.compare(threshold=6)           # 秒出结果
```

查一张不在索引里的图：

```python
for path, distance in index.nearest("new-photo.jpg", k=5):
    print(distance, path)
```

## 阈值怎么选

`threshold` 是两张图的 64 位指纹之间**允许有多少位不同**：

| 取值 | 效果 |
|---|---|
| 0–4 | 字节级相同的副本、重新编码、只改了元数据 |
| **10（默认）** | 改过尺寸、重新压缩、调过亮度、转成黑白 |
| 14–18 | 同一素材的裁切版或轻微重构图 |
| >20 | 开始把真正不同的照片并到一起 |

截图和数字艺术通常用 6–8；手机相册因为传感器噪声和 HDR 处理，得用 10–14。
**先拿 200 张样本调准再跑全量**——不同来源的最佳阈值能相差一倍。

## 算法原理

每张图四步，纯 numpy，不依赖 OpenCV：

1. **缩到 32×32 灰度** — 抹掉分辨率和压缩差异
2. **二维 DCT，取左上 8×8** — 低频系数描述的是版面布局，不是像素
3. **每格与中位数比较** — 高于记 1，低于记 0，得到 64 位
4. **距离 = 异或后数 1 的个数** — 两个指纹一次 XOR，数一下有几个 1

同时算两个哈希并取平均：`pHash`（DCT，抗缩放和重压缩）和 `dHash`（水平梯度，
对真实结构改动敏感）。取平均能压掉单独用任何一个都会出现的假阳性。

### 颜色否决

用 8×8 网格的区域均值取 192 维颜色签名，在 HSL 空间比较：

```
distance = 0.55·Δ色相·minSat + 0.35·Δ饱和·minSat + 0.90·Δ亮度
```

`minSat`（两者饱和度的较小值）是关键的。转灰度会毁掉饱和度，但不会改变感知亮度；
把色相和饱和度两项乘以较小饱和度，灰度图的这两项会自动归零，只比较亮度。
于是「原图 vs 它的黑白版」能通过，而「红苹果 vs 绿苹果」（两者都高饱和）会被全强度拦住。

只取 64 个网格里差异最大的 12 个求均值，所以改掉角落里一个小物体的颜色不会被
剩下 52 个没变的格子稀释掉。

实测分离度：同图变体 **0.002–0.06**，无关图 **0.39–0.90**，默认阈值 0.18 正好在空档里。

### 低信息量守卫

pHash 只有在图像存在低频结构时才有意义。纯色、线稿、规则条纹这类图的交流系数接近 0，
中位数比较退化，所有这种图都哈希到同一个值。`imgsim` 会测 AC/DC 能量比，
低于 0.02 时要求颜色几乎一致（0.06 以内）才判为相似。

## 性能

4 核 Windows 笔记本实测，1000 张 400×400 JPEG：

| 阶段 | 耗时 |
|---|---|
| 解码 + 指纹 | 1.7 s |
| 相似度矩阵 + 聚类 | 2.3 s |
| **合计** | **4.0 s** |

两个关键优化：`Image.draft()` 让 libjpeg 在解码阶段就降采样（千万像素照片能快 5–10 倍）；
汉明距离走 `numpy.bitwise_count`，即硬件 popcount，作用在 uint64 上。

内存占用是每张图一个指纹（8 字节哈希 + 192 个 float 的颜色签名），一百万张图也能放进内存。

超过约 5 万张后，稠密的 N×N 矩阵成为瓶颈（5 万张 → 2.5 GB）。这时改用
`index.iter_pairs()` 流式处理，或先按哈希前缀分桶再两两比较。API 给这两种做法都留了口子。

## 语义模式

感知哈希回答的是「这是不是同一张图」，它对**含义**完全无感：同一片海滩从不同角度拍的
两张照片，哈希结果天差地别。

```bash
imgsim ./photos --mode embed --threshold 0.15
```

`embed` 模式提取 512 维 ResNet18 特征算余弦距离，能认出不同视角下的同一主体，
代价是依赖 PyTorch，且细粒度区分能力弱得多——它会把两场不同的日落合并在一起。
需要「主题相似」而不是「同一文件」时才用它。

## API 一览

| 成员 | 用途 |
|---|---|
| `ImageIndex(mode="hash")` | 创建索引 |
| `.add(path)` / `.add_many(paths)` / `.add_dir(dir)` | 索引图片（并行解码） |
| `.compare(threshold, color_threshold, pairs_limit)` | 分组，返回 `Result` |
| `.nearest(path, k)` | 任意图片的 K 近邻 |
| `.save(path)` / `ImageIndex.load(path)` | 保存 / 恢复指纹 |
| `.iter_pairs(threshold)` | 流式产出相似对，不占内存 |
| `.failed` | 解码失败的路径 |
| `Result.groups` / `.pairs` / `.duplicates` | 结果 |
| `render_html(result)` / `save_html(result, path)` | 报告 |

## 开发

```bash
git clone https://github.com/zhouchuan315-png/imgsim
cd imgsim
pip install -e ".[dev]"
pytest
```

测试用固定随机种子现场生成图片，仓库里不放二进制素材。

## 许可证

MIT，见 [LICENSE](LICENSE)。

English: [README.md](README.md)
