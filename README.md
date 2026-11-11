# VITS Studio

本地中文 VITS 音色训练工作台：上传录音 → 自动转写和切分 → 校对 → 微调 → ONNX 导出 → 文本试听。

## 安装与启动

先克隆仓库，或下载源码压缩包并解压。以下命令在包含 `app.py` 和 `install.sh` 的项目根目录执行。首次安装需要联网，首次使用转写或推荐底模时还需下载对应模型。

### Windows

1. 安装 Miniconda 或 Anaconda。打开 **Anaconda Prompt**，执行 `cd /d "你的项目目录"`。
2. 创建并激活 Python 3.12 环境，然后安装 FFmpeg 和 Python 依赖。已有名为 `py312` 的 Python 3.12 环境时，从激活命令开始。

   ```bat
   conda create -n py312 python=3.12 pip -y
   conda activate py312
   conda install -c conda-forge ffmpeg -y
   python -m pip install -r requirements.txt
   ```

3. 检查版本，再启动应用。

   ```bat
   python --version
   ffmpeg -version
   ffprobe -version
   python app.py
   ```

4. 保持终端打开，在浏览器访问 <http://127.0.0.1:7860>。按 Ctrl+C 停止。如果端口被占用，运行 `python app.py --port 7861`，再访问 <http://127.0.0.1:7861>。以后启动只需进入项目目录，激活 `py312` 后运行 `python app.py`。

找不到 `conda` 时，检查是否打开了 Anaconda Prompt。提示缺少 Python 模块时，在激活 `py312` 后重新运行第 2 步的 pip 命令。

### Linux x86_64

1. 将源码放在当前用户有写权限的目录。系统需有 Bash、`curl`、`tar`、`flock` 和支持 `hostname -I` 的 `hostname`。使用 NVIDIA GPU 时，宿主机需安装 NVIDIA 驱动。
2. 运行安装脚本。它会下载 micromamba，按锁定清单安装 Python 3.12、FFmpeg、PyTorch 等依赖到 `runtime/`，生成缺失的 `deploy.env` 并检查运行环境。

   ```bash
   bash install.sh
   ```

3. 看到 `Installed entirely under ...` 后，启动并检查服务。

   ```bash
   bash start.sh
   bash status.sh
   ```

4. 在服务器本机访问 <http://127.0.0.1:7860>，或在局域网访问 `http://服务器IP:7860`。修改端口、允许访问的主机名或 GPU 设置时，编辑 `deploy.env`，然后依次运行 `bash stop.sh` 和 `bash start.sh`。

启动失败时，运行 `tail -n 50 data/logs/web.log` 查看错误。更新依赖前，先执行 `bash stop.sh`，再运行安装和启动命令。更多配置见 [Linux 部署说明](DEPLOYMENT.md)。

## 训练一个声音

1. 新建音色项目，上传单人中文录音。支持 WAV、MP3、FLAC、M4A、OGG、WebM、MP4、MKV。
2. 选择 Whisper 模型，点击「自动切分并转写」。初次使用需要下载模型；Small 质量和速度较均衡，CPU 可选择 Tiny/Base。按词语时间戳切分，最长约 12 秒。
3. 逐段试听并修改文字，取消勾选有噪声、多人讲话或识别错误的片段，保存校对。
4. 点击「下载推荐底模」（约 331 MB，完成后自动填入路径），或准备 **VITS-fast-fine-tuning 配套的 PyTorch 底模**，目录必须包含 `config.json`、`G_0.pth`、`D_0.pth`。配置必须带 `symbols`、`data`、`train`、`model`，且声学结构与权重匹配。填写底模目录。
5. 点击「读取底模音色」，选择训练模式和参考/目标音色，再设置本次训练轮数、批次和学习率，点击「开始训练」。自动选择 CUDA 或 CPU。至少需要两个有效片段，其中一个留出、不参与训练；建议准备 3 分钟以上录音。
6. 每轮结束保存检查点；停止任务将丢失当前尚未完成的一轮。保留「继续训练」可恢复生成器、判别器和优化器。本次轮数会追加到已完成轮数上。
7. 到「试听与导出」转换模型。程序生成 ONNX、符号表、中文词典和 FST，检查不同文本长度能否推理，再验证 sherpa-onnx 加载并打成 `.tar.bz2`。
8. 选择新模型，输入中文文本，点击「生成试听」。页面内播放或下载 WAV，也可下载模型包。

训练模式：

| 模式 | 参考/目标音色 | 输出 |
|---|---|---|
| 新增音色（新项目默认） | 复制参考音色的嵌入作为初始值 | 保留原有编号，新增编号为原音色总数 |
| 修改已有音色 | 选择要更新的音色编号 | 保留数量，更新所选编号的音色名称 |
| 仅输出一个音色 | 选择参考音色作为初始值 | 单音色模型，目标编号为 0 |

新增/修改默认开启「保护其他音色」：冻结生成器共享参数，只训练目标音色嵌入，其他音色的参数保持不变，拟合能力相应受限。取消保护会完整微调生成器，其他音色可能变化。单音色模式始终完整微调。所有模式都另存结果，不修改原始底模。导出保留全部输出音色编号与名称，试听时可选择音色。

同一检查点续训要求底模路径、训练模式、参考/目标音色、参数保护设置一致；切换时请另建项目或取消「继续训练」（下一轮保存将覆盖该项目的旧检查点）。旧版单音色检查点按「仅输出一个音色、参考编号 0」继续训练。

不训练也可导出底模：填写底模目录，点击「导出底模为 tar.bz2」。只需要 `config.json` 和 `G_0.pth`，无需项目录音或 `D_0.pth`。程序保留全部底模音色，转换 ONNX、补齐词典并验证推理，完成后显示直接下载链接，也可在「试听与导出」的模型库中下载。可选同时生成 INT8 模型。上游配置中 `n_speakers=0` 时，自动从生成器的说话人嵌入读取实际音色数量；没有名称表时显示编号。

底模来源：[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)
和 [上游 WebUI](https://github.com/ANONEKO777/sherpa-onnx-vits-fine-tuning-webui/tree/master/training_data/pretrained_models)。
ONNX 语音包不能直接反向用作训练底模。只加载可信的 `.pth` 文件。

本机当前是 CPU 版 PyTorch。CPU 支持真实训练，但正常尺寸底模会明显慢于 GPU。
训练完成不代表音质达标；音色相似度取决于底模、录音质量、数据量和训练轮数。

## 从抖音视频链接导入素材

在上传区粘贴 `https://www.douyin.com/video/…` 、`https://v.douyin.com/…` 或 `https://…douyinvod.com/…` 媒体直链，
点击「从链接导入音频」。支持含链接的分享文字。后台使用 yt-dlp 下载单个视频，
再用 FFmpeg 提取为 16 kHz 单声道 WAV，加入当前音色的素材列表；之后点击自动切分并转写。
解析、下载字节数/百分比、提取音频和失败状态会显示在页面与任务日志中。

抖音可能要求近期 Cookies。遇到 403 或 Cookies 提示，在「访问验证设置」填写本地
Netscape 格式 `.txt` Cookies 文件路径。程序使用临时副本，不修改原文件，也不调用第三方解析服务。
Cookie 不保证解决平台限制；无法下载时，可在浏览器保存视频后直接上传。
当前接受抖音单个视频、短链接和 douyinvod.com 媒体直链，不支持主页、直播或合集，文件上限 1 GB。
直链必须保留全部查询参数。下载后可在进度区下载原素材；若直链仅含画面，没有音轨，会提示补充音频链接或完整视频。

## 导入已有模型

「试听与导出」支持填写本地 VITS 模型目录或 `.tar.bz2` 语音包路径，例如：

```text
C:\yico\语音包\vits-zh-hf-bronya.tar.bz2
C:\yico\语音包\vits-piper-zh_CN-huayan-medium.tar.bz2
```

导入时会验证 ONNX 的 VITS 元数据及 sherpa-onnx 可加载性。支持中文 VITS 和 Piper；
Kokoro、Matcha、Supertonic 等不同架构尚未接入。说话人编号依模型而定，单音色训练结果使用 ID 0。
中文模型不一定支持英文；不在词典中的内容可能被跳过。

## 文件与日志

```text
data/
  projects/<id>/project.json     # 音色项目、录音清单与校对文本
  projects/<id>/uploads/         # 原始上传录音
  projects/<id>/clips/           # 切分的 WAV
  projects/<id>/training/        # config、G_latest、D_latest、resume.pth
  models/<id>/                  # 可部署 ONNX 资源与 voice.tar.bz2
  jobs/<id>/                    # 任务状态、持续日志、试听 WAV
  whisper/                      # Whisper 模型缓存
```

训练与转写在后台独立进程执行；同一时间只允许一个任务，以免抢占资源。
页面刷新不会停止任务，服务重启后未完成任务显示「已中断」。不提供外网访问或多人隔离。
原始录音不会上传给模型服务，首次下载依赖或模型需要联网。

## 开发验证

```powershell
conda run -n py312 python -m unittest discover -s tests -v
```

测试覆盖项目持久化、上传回滚、任务恢复、请求来源、压缩包路径校验，以及
Numba 对齐算法与穷举最优路径的一致性、动态相对位置嵌入。
本机另外验证了小型随机模型真实训练、断点续训、检查点保存、ONNX 动态输入导出、INT8 量化、
已有中文包真实合成，以及 Tiny Whisper 实际转写和切分。
随机模型仅用于流程验证，不是可用的音色底模。

API 文档：http://127.0.0.1:7860/api/docs

源码与适配说明见 `vendor/README.md`。

## 删除与恢复

声音训练页可删除整个音色、单个上传素材或单个切分片段。删除素材时会同时移走它关联的片段；
删除音色时会移走该项目的素材、片段和训练检查点，已导出的模型仍保留在模型库中。
删除内容移入 `data/trash/<删除编号>/`，不会立即清除。点击「撤销上次删除」可恢复，刷新后仍可撤销。
恢复会保留已校对文字、片段启用状态和训练检查点。任务运行中不能删除。
新切分片段记录原始素材编号，旧片段按唯一文件名关联；旧素材重名且无法确定来源时会要求先处理关联片段。

试听与导出页的模型库支持删除导入的语音包，操作只移走 WebUI 内的副本，不修改导入源文件；可使用‘撤销语音包删除’恢复。任务运行期间禁止删除和恢复。

## Bilibili 素材

视频链接导入同时支持 `bilibili.com/video/BV…/`、`av…` 和 `b23.tv` 短链接。
可粘贴带跟踪参数的分享链接，程序会保留 `p` 分 P 编号并去掉其他跟踪参数。
B 站优先下载音轨，再转换为训练使用的 16 kHz 单声道 WAV；未填写分 P 时仅处理单个视频的默认分 P。
如遇登录或风控限制，可在访问验证设置中提供对应平台的 Netscape Cookies 文件。

## 分离人声后切分

勾选「先分离人声，再切分转写」时，使用 Demucs htdemucs 分离人声后再生成训练片段。原素材保持不变；首次加载需下载分离模型到 `cache/torch/checkpoints/`。默认使用可用 CUDA，未提供 CUDA 时使用 CPU。分离可能残留背景音乐或产生失真，请试听校对。

「只切分新增素材」保留已有片段及校对；若要清理已有片段的背景音乐，请选「全部重新切分」，这会替换已有片段及校对文字。
