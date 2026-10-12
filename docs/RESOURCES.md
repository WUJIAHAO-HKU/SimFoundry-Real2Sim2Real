# 资源清单与在线安装 / Resource inventory

更新日期：2026-10-12。适用于本分支的 Linux x86_64 / Python 3.11 / CUDA 12.8 安装组合。
项目源代码和文档素材保存在 Git/Git LFS；模型、仿真资产、环境、缓存与实验输出从下面的来源获取或重新生成。

## 1. 全新机器安装，不依赖英博云目录

准备 NVIDIA 驱动、Miniforge/Mamba、Git LFS、curl、tar 和编译工具，然后运行：

```bash
git lfs install
git clone https://github.com/WUJIAHAO-HKU/SimFoundry-Real2Sim2Real.git
cd SimFoundry-Real2Sim2Real
bash scripts/installation/install_everything.sh --only "simfoundry hunyuan any6d da3"
```

这会创建环境并下载软件及部分仿真资产，安装体积较大。GPU 环境名可以通过各安装器的
`--env-name` 参数指定；安装器使用 Mamba 的环境列表检测环境，不再检查某个服务器上的固定目录。
完整步骤与硬件要求见 [INSTALL.md](INSTALL.md)。OpenAI 接口配置见 [OPENAI_VLM.md](OPENAI_VLM.md)。

四个核心安装器共用 [python_packages.sh](../scripts/installation/python_packages.sh)：

- PyTorch `2.7.0+cu128`、torchvision `0.22.0+cu128`、torchaudio `2.7.0+cu128`、
  xformers `0.0.30`：从 [PyTorch CUDA 12.8 wheel 索引](https://download.pytorch.org/whl/cu128/)安装，
  同时解析运行库依赖。版本组合参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。
- Pillow `11.3.0`：从 [PyPI](https://pypi.org/project/pillow/11.3.0/)安装。
- FlashAttention `2.8.3`：从 [PyPI](https://pypi.org/project/flash-attn/2.8.3/)安装。
  上游构建过程可能下载匹配的预编译包；没有匹配包时需要本机 nvcc、编译器和足够的系统内存。
- gsplat：下载 [固定源码提交](https://github.com/nerfstudio-project/gsplat/tree/0b4dddf04cb687367602c01196913cde6a743d70)，
  默认缓存到 `${XDG_CACHE_HOME:-$HOME/.cache}/simfoundry/sources`。下载完成且压缩包检查通过后才保存为有效缓存。
- CUDA Toolkit 从 NVIDIA Conda channel 获取；FAISS GPU `1.12` 从 PyTorch Conda channel 获取。
  其他依赖及其源码提交仍由各 `install_*.sh` 固定。

本组合固定 CUDA 12.8；更改 `--cuda-version` 不能自动产生另一套兼容的 torch/xformers 配对，
共享安装函数会明确拒绝不匹配的 CUDA 版本。官方 Linux wheel 要求兼容的 glibc（所选 torch wheel 为 manylinux_2_28）。

### 可选缓存与镜像

以下变量均可不设置。缓存不完整时仍允许在线获取缺失依赖，不会强制 `--no-index`：

```bash
# 使用自己的缓存位置；路径可以包含空格。
export SIMFOUNDRY_WHEEL_DIR=/path/to/wheels
export SIMFOUNDRY_SOURCE_CACHE=/path/to/source-archives

# 只有确认 Python / torch / CUDA / C++ ABI 全部匹配时才指定此项。
export SIMFOUNDRY_FLASH_ATTN_WHEEL=/path/to/compatible_flash_attn.whl

# 可选覆盖官方索引；所用镜像必须提供相同版本和 CUDA 变体。
export SIMFOUNDRY_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
export SIMFOUNDRY_PYPI_INDEX_URL=https://pypi.org/simple
```

这些变量只控制共享的 torch/Pillow/FlashAttention 安装步骤，不会把所有第三方安装器的网络请求重定向。
系统盘空间有限时，还可设置标准 `PIP_CACHE_DIR` 和 `TMPDIR` 到自己的大容量可写磁盘；
即使 `pip --dry-run` 也可能下载依赖 wheel，不等于零磁盘占用。
不设置 `SIMFOUNDRY_WHEEL_DIR` 时，不会尝试读取 `/root/data/SOA/wheels`。

## 2. 核心模型与版本记录

首次使用 Hugging Face 前运行 `hf auth login`。SAM3、DINOv3 和 RMBG 的访问条件以各模型页面为准，
账号必须先获得对应权限；公开 Git 仓库不提供模型访问凭据。

- **SAM3：** [facebook/sam3](https://huggingface.co/facebook/sam3)，分割与目标掩码。
  核心文件 `sam3.pt`；默认由上游构建器获取，也可将
  `SIMFOUNDRY_SAM3_CHECKPOINT` 指向本地文件。
- **Depth Anything 3：** [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)，
  当前深度模型默认版本为 **1.1**。下载整个模型目录；`SIMFOUNDRY_DA3_CHECKPOINT` 可指定该目录。
  旧 `DA3NESTED-GIANT-LARGE` 仍是代码支持的可选模型，不能与 1.1 混为一份权重。
- **Hunyuan3D 2.1：** [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)，
  形状需要 `hunyuan3d-dit-v2-1` 和 `hunyuan3d-vae-v2-1`，纹理需要 `hunyuan3d-paintpbr-v2-1`。
  不要只下载主模型而遗漏 VAE、纹理配置和编码器。形状加载器使用上游下载逻辑；
  `SIMFOUNDRY_HUNYUAN_CHECKPOINT` 当前只覆盖纹理模型的来源，不会覆盖形状加载器。
- **DINOv2 giant：** [facebook/dinov2-giant](https://huggingface.co/facebook/dinov2-giant)，Hunyuan 纹理特征。
  下载模型与配置；`SIMFOUNDRY_DINOV2_CHECKPOINT` 可指定本地目录。
- **RMBG 2.0：** [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)，根据所选分割/去背景实现获取
  Transformers 权重或 `onnx/model.onnx`。两种格式不是同一个文件；ONNX rembg 缓存文件名为 `bria-rmbg.onnx`。
- **U2Net：** [rembg 官方发布文件](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx)，
  缓存到 `${U2NET_HOME:-$HOME/.u2net}/u2net.onnx`；用于选择 U2Net 的去背景分支。

[resources.huggingface.json](resources.huggingface.json) 记录核验当天上述模型及其他辅助模型的官方
repository ID、不可变 commit、访问状态和文件列表。下面的命令从清单读取版本，可避免复制错 commit：

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
revision() {
  python -c 'import json,sys; d=json.load(open("docs/resources.huggingface.json")); print(next(m["revision"] for m in d["models"] if m["repo_id"]==sys.argv[1]))' "$1"
}

export SIMFOUNDRY_SAM3_CHECKPOINT="$(hf download facebook/sam3 sam3.pt \
  --revision "$(revision facebook/sam3)")"
export SIMFOUNDRY_DA3_CHECKPOINT="$(hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --revision "$(revision depth-anything/DA3NESTED-GIANT-LARGE-1.1)")"
export SIMFOUNDRY_DINOV2_CHECKPOINT="$(hf download facebook/dinov2-giant \
  --revision "$(revision facebook/dinov2-giant)")"
export SIMFOUNDRY_HUNYUAN_CHECKPOINT="$(hf download tencent/Hunyuan3D-2.1 \
  --revision "$(revision tencent/Hunyuan3D-2.1)")"
```

这些导出变量只对当前 shell 及其子进程生效。未指定本地路径的上游加载器仍可能访问其默认分支；
以上预下载命令不是“整个推理流水线已强制离线、所有模型都已锁定”的保证。

### 英博云实际下载版本

[resources.ebcloud.json](resources.ebcloud.json) 保存六组已下载模型（46 个文件）及辅助权重的来源、
逐文件 revision、大小和 SHA-256。六组模型来自当时使用的 ModelScope 下载记录：

- `facebook/sam3`
- `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
- `Tencent-Hunyuan/Hunyuan3D-2.1`
- `facebook/dinov2-giant`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `briaai/RMBG-2.0`

清单同时给出每个文件绑定 revision 的下载 URL。对应来源页面为 `https://modelscope.cn/models/<repo_id>`。镜像的逐文件 revision 不等于 Hugging Face
commit；**两个 JSON 分别记录历史已下载文件与今日官方来源，不宣称它们逐字节相同。**
辅助文件哈希是本地实际下载字节的记录，不冒充上游发布的校验值。校验下载文件可使用 `sha256sum`。
英博云目录只是历史位置，新的在线安装不需要复制这些缓存目录。

## 3. 位姿、立体深度与辅助权重

以下来源已纳入 [download_checkpoints.sh](../scripts/installation/download_checkpoints.sh)：

- **FoundationPose refiner：** [Google Drive 目录](https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC)，
  下载 `model_best.pth` 和 `config.yml` 到 `deps/FoundationPose/weights/2023-10-28-18-33-37/`。
- **FoundationPose scorer：** [Google Drive 目录](https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj)，
  下载权重和配置到 `deps/FoundationPose/weights/2024-01-11-20-02-45/`。
- **FoundationStereo：** [Google Drive 目录](https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-)，
  保存到 `deps/FoundationStereo/pretrained_models/23-51-11/`。DA3 单目深度路线不要求切换为这个模型。
- **SAM2.1：** [官方下载文件](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)，
  保存到 `checkpoints/sam2.1_hiera_large.pt`，供相关可选工具使用。
- **DepthPro：** [Apple 官方权重](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt)，
  保存到 `deps/ml-depth-pro/checkpoints/depth_pro.pt`。
- **RealESRGAN x4plus：** [官方发布文件](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)，
  保存到 `deps/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth`。

在环境与源码依赖安装完成后运行：

```bash
bash scripts/installation/download_checkpoints.sh --default
```

注意：该旧下载器还会尝试下面的 VOID 大模型；并非仅下载核心 DA3/Hunyuan 模型。
它也不是上述 Hugging Face 模型预下载命令的替代品。Google Drive 可能有下载配额，失败会返回非零状态。

## 4. 可选分支的模型

- 自动背景： [alibaba-pai/CogVideoX-Fun-V1.5-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP)
  和 [netflix/void-model](https://huggingface.co/netflix/void-model)。下载器将基础模型目录及
  `void_pass1.safetensors`、`void_pass2.safetensors` 放在 `deps/void-model/`，需要相应访问权限。
- 其他深度/匹配分支：[Rain729/Prior-Depth-Anything](https://huggingface.co/Rain729/Prior-Depth-Anything)、
  [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)。
  官方 revision 在 `resources.huggingface.json` 中。
- TRELLIS：[microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large)；
  TRELLIS.2：[microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)，安装说明见 `INSTALL.md`。
- Pixal3D：[TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D)、
  [Ruicheng/moge-2-vitl](https://huggingface.co/Ruicheng/moge-2-vitl)、
  [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)。
  使用 `install_pixal3d.sh` 中已固定的模型 revision、NAF 权重链接和哈希；
  不要用本清单核验当天的 DINOv3 revision 擅自替代该可选后端自己的固定版本。
- FLUX 可选图像生成：[black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)。
  默认 OpenAI 图像接口不需要本地 FLUX 权重。

可选模型并非全部在英博云安装或测试过；其上游许可、账号授权和资源需求分别适用。
本清单不包含训练 UniDex 所用数据和 checkpoint，那些属于另一个 SOA 模型工作区。

## 5. 视频、场景与机器人资产

- **已随 Git 提供的示例视频：** `docs/assets/example_videos/`，例如 `Fruits.mp4`；无需另找下载地址。
  PNG/GIF 文档素材由 Git LFS 提供，已有克隆可以运行 `git lfs pull`。
- **公开示例场景：** [nadunRanawaka1/simfoundry-assets](https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets)。
  在项目根目录运行 `hf download nadunRanawaka1/simfoundry-assets --repo-type dataset --local-dir assets`。
  两个数据集核验当天的 revision 记录在 `resources.huggingface.json` 的 `datasets` 字段，
  需要固定版本时将相应 `--revision` 追加到下载命令。
- **机器人资产：** [behavior-1k/omnigibson-robot-assets](https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets)。
  `install_simfoundry.sh` 调用 OmniGibson 下载器，另补充 `models/franka/franka_robotiq/` 子目录；
  目标位置是 `deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/`。不要仅下载 robotiq 子目录而遗漏基础资产。
- **BEHAVIOR 对象资产与仿真安装：** 由固定版本的
  [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) 安装器处理。
  按所选任务获取需要的数据，不把整个 BEHAVIOR 数据集作为每个重建任务的必选下载。
- **YAM 机器人：** 安装器从 [ARISE-Initiative/yamlab 固定提交](https://github.com/ARISE-Initiative/yamlab/tree/ec0455d2b4ce35f21fc126418ea5e74ac567133d)
  获取 USD，并转换到 `models/yam/usd/yam.usda`。

## 6. 未发布的英博云产物

`data/fruits_preparation_smoke/` 中的重建网格、USD、场景 JSON、视频，以及 `runs/` 中的日志、
API 响应缓存均是本次运行生成的产物，**目前没有公开下载地址**。它们不是缺失的开源模型下载链接。
需要通过重建流水线重新生成，或另行发布带完整资产引用的场景包。
既有实验说明见 `OPENAI_VLM.md`；其中英博云专用 launcher 路径不是这个仓库自带的安装入口。

环境目录通过安装器重建；访问令牌和 API 密钥由使用者自行配置。不要将 `api_keys.txt`、HF token、
代理配置或 Tailscale 状态加入资源包。

## 7. 本次验证边界

本次核验官方 PyTorch wheel 索引、核心 HF 仓库 metadata/commit，以及英博云已有下载记录；
新增测试覆盖无缓存、部分缓存、在线命令、错误配置与下载失败不污染源码缓存。
官方 CUDA 12.8 源的 torch、torchvision、torchaudio、xformers 四个核心 wheel 已通过
`--dry-run --no-deps` 下载检查；Pillow 的独立 PyPI 检查超时，未完成。
完整依赖 dry-run 因系统盘空间中止；不能据此宣称完整依赖解析或全新环境安装已经通过。
没有在现有英博云训练/仿真环境上重跑整套安装，也没有重新下载全部大模型或在全新机器上验证整条重建流水线。
