# goovo·human — 口播数字人智能体

替你口播的数字人主播，主要面向**微信视频号直播**（也支持任何 RTMP 平台）。
在 Apple Silicon Mac 上即可实时运行；有 Apple GPU 时自动启用神经口型（MuseTalk MLX），无需 NVIDIA 显卡。

```
口播稿 / LLM 撰稿 ──► 神经语音合成(TTS) ──► 音视频实时合成 ──► 视频号 RTMP 推流
   scripts/*.md          edge-tts/克隆          双视频数字人        或 本机预览/录制
        ▲                                        ▲
        └────────── Web 控制面板（插播/暂停/切换模式）┘
```

**数字人原理**：录制你本人两段视频——闭嘴待机 `idle` 与自然说话 `talk`。
- **neural 引擎（默认优先，Apple GPU）**：MuseTalk 1.5 的 MLX 移植版按**音频逐帧生成**你本人的下半脸
  （音素级唇形，圆唇/展唇/闭口都是真实形状），预渲染缓存后播放时按头部姿态对齐贴回 idle 画面。
- **mouth 引擎（CPU 兜底）**：从 `talk` 提取 a/o/i/u 口型素材，按 TTS 字级时间戳 + 拼音韵母 + 音频包络
  驱动嘴部连续形变。零模型推理，任何 Mac 都能实时。
- 无口型素材时自动退化为双视频切换（`engine = "switch"`）。

## 快速上手

仓库只含源码与配置（约 0.7MB）：虚拟环境、模型权重、口型/语音缓存、录制输出和个人素材
均被 `.gitignore` 排除，按下面步骤在本机生成即可。

```bash
uv sync                          # 安装依赖（需要 uv、ffmpeg：brew install ffmpeg）
uv run goovo placeholder         # 生成示例形象视频（先跑通流程）
uv run goovo prepare-mouth       # 录制自己的素材后：提取口型（启用音频驱动口型）
uv run goovo prepare-neural      # 可选：预计算神经口型 latent（首次 run 也会自动做）
uv run goovo warmup              # 预合成音频 + 预渲染神经口型（首次约 10 分钟，之后开播即播）
uv run goovo doctor              # 环境自检
uv run goovo tts "大家好，欢迎来到直播间"   # 试听语音 → output/tts_test.wav
uv run goovo run                 # 启动！默认本机预览窗口 + 控制面板
```

打开 http://127.0.0.1:8765 即可看到控制台：实时状态、输入文本让数字人**立刻插播**、暂停/恢复、切换撰稿模式。

```bash
uv run goovo run --sink record   # 不弹窗，录制到 output/live.mkv（验收/试播）
uv run goovo say "欢迎新进直播间的朋友" -i   # 命令行插播（-i 打断当前）
```

## 录制你自己的形象（关键一步）

占位视频只是演示，正式开播前请录制：

1. **竖屏 9:16**（手机后置摄像头，1080×1920 或更高），三脚架固定机位，光线充足均匀。
2. `assets/idle.mp4`：**闭嘴静默** 10–30 秒。自然微笑、偶尔眨眼、轻微呼吸起伏，不要大幅动作。
3. `assets/talk.mp4`：**自然说话** 10–30 秒。内容随意（念段新闻即可），口型自然开合，手势幅度小。
4. 两段视频**同一机位、同一景别、同一光线**，首尾姿态尽量接近，循环播放才不跳。
5. 覆盖 `config.toml` 里 `[avatar]` 的 `idle` / `talk` 路径即可（程序自动缩放裁切到输出分辨率）。

> 只有一段视频？把 `talk` 留空或指向同一文件，退化为单视频循环 + 口播音频。

## 神经口型（neural 引擎，推荐）

MuseTalk 1.5 MLX 版跑在 **Apple GPU（Metal）** 上：M2 Pro 的 19 核 GPU 可用，不需要 NVIDIA 显卡。
口型由音频直接生成，形状是真人说话的形状，而非贴图拉伸。
M2 Pro 实测渲染约 6 帧/秒（256² 脸、batch 8、fp16），**不够边播边渲染**，所以采用预渲染缓存：
开播前把循环稿件逐帧渲染好（`goovo warmup` 或首次 `run` 自动做，约 100 秒/20 秒音频），
之后播放零 GPU 开销、音画逐帧同步；缓存按音频内容哈希键控，稿件不变就永久命中。

```bash
# 1) 独立渲染环境（与主环境隔离，避免依赖冲突）
uv venv .venv-muse --python 3.12
uv pip install --python .venv-muse/bin/python "fusion-mlx[video]>=0.10.2,<0.11" mediapipe
# 2) 权重（whisper 编码器走 hf-mirror；UNet/VAE 走 ModelScope 后转 MLX fp16）
FETCH_THREADS=16 .venv/bin/python tools/fetch_hf.py mlx-community/MuseTalk-1.5-fp16 \
    .cache/models/musetalk-mlx-fp16 config.json whisper_encoder.safetensors
FETCH_SOURCE=modelscope FETCH_THREADS=12 .venv/bin/python tools/fetch_hf.py AI-ModelScope/MuseTalk \
    .cache/models/musetalk-torch musetalkV15/unet.pth
FETCH_SOURCE=modelscope .venv/bin/python tools/fetch_hf.py stabilityai/sd-vae-ft-mse \
    .cache/models/musetalk-torch/sd-vae-ft-mse diffusion_pytorch_model.safetensors
uv pip install --python .venv-muse/bin/python torch   # 仅转换时用
.venv-muse/bin/python tools/convert_musetalk.py
# 3) 预计算 idle 逐帧 latent（换 idle 视频后需重跑；run 时也会自动检测）
uv run goovo prepare-neural
uv run goovo run                 # engine=auto 自动优先 neural
```

- 渲染 worker：`tools/musetalk_worker.py`（`.venv-muse`），日志 `.cache/musetalk-worker.log`；
  缓存 `.cache/mouth/<音频哈希>-<idle 版本>/f_*.jpg`。换 idle 视频后缓存自动失效重渲。
- 播放对齐：缓存帧按当前帧头部三关键点仿射对齐（平移/缩放/倾斜跟随）+ 椭圆羽化 + 肤色补偿。
- 新稿件（LLM 撰稿/面板插播）首遍用 mouth 引擎兜底、后台渲染，循环第二遍起自动 neural。
- worker 缺失/崩溃自动回退 mouth 引擎；`[avatar].engine` 可强制 `neural / mouth / switch / loop`。
- 离线对比样片：`.venv-muse/bin/python tools/musetalk_worker.py proto <wav> <out.mkv> [秒数]`。

## 口型同步（mouth 引擎，CPU 兜底）

```bash
uv run goovo prepare-mouth       # 从 idle 定位嘴部、从 talk 提取四类口型 → assets/mouth/
uv run goovo run                 # engine=auto 时自动启用 mouth 引擎
```

- **何时做什么口型**：edge-tts 返回字级时间戳，拼音韵母映射到 a/o/i/u 口型（缓存于 `.cache/tts/*.json`）；克隆音频可用 `.venv-clone/bin/python tools/align_cache.py` 生成字级时间戳。
- **开合跟随音频**：口型开合由 10ms 音频包络驱动（快起音、慢释放），不再只靠固定贴图切换。
- **开多大 / 何时闭嘴**：音频活动门限带迟滞（起音 ~35ms、静音保持 ~140ms、释放 ~180ms），音节间隙不逐字闪闭；强度只做浅调制，避免嘴部按音节上下抽动。
- **自然过渡**：口型贴图先统一到 idle 参考姿态，再做 ~90ms 交叉淡化；包络 ~50ms 平滑，逐帧连续渐变而非硬切（消除"闪烁感"和机械振动）。
- **贴得稳（连续形变）**：说话时逐帧半分辨率跟踪嘴角×2+鼻尖三关键点（~80ms 平滑），嘴部区域按音频开合做连续网格形变（下唇/下颌下拉、嘴角按韵母拉伸/收圆），头部平移/转动/俯仰都跟随；口内纹理椭圆羽化 + 肤色补偿。
- 素材或效果不满意：重录 `talk` 后重跑 `prepare-mouth`；`[avatar].engine` 可强制 `mouth / switch / loop`。
- 局限：口型为韵母级近似（非音素级神经唇形），安静辅音处靠包络收口。升级见文末。

## 微信视频号开播

### 路线 A：Mac 微信直播伴侣（推荐：实名认证即可，无粉丝门槛）

官方 Mac 版微信内置「视频号直播伴侣」，支持**窗口 / 屏幕画面源**，正好捕获本项目的预览窗口：

1. 视频号助手 channels.weixin.qq.com → 直播 → 直播管理 →「直播方式」模块 →「微信 macOS 版直播」下载，
   安装支持直播伴侣的 Mac 微信（提示替换现有微信时选"替换"）。
2. 系统设置 → 隐私与安全性 → **录屏与系统录音** → 允许微信。
3. 启动数字人：`uv run goovo run`，出现「goovo-human 直播预览」窗口。
4. Mac 微信 → 更多 → 视频号直播伴侣 → 添加画面源 → **窗口** → 选「goovo-human 直播预览」，右键"等比例适配全屏"。
5. 声音：看开播页音频电平表是否随口播跳动（伴侣需采集系统声音）。若你的版本窗口源不带声音，用下面的 BlackHole 路由。
6. 填封面 / 标题 / 分类 → 开始。Mac 版可见范围仅"公开"，**务必先 `--sink record` 试播验收，再正式开播**。

**声音兜底（BlackHole 路由）**：`brew install blackhole-2ch` →「音频 MIDI 设置」新建多输出设备 = BlackHole 2ch + 扬声器 →
系统输出选该多输出设备 → 直播伴侣的麦克风输入选 BlackHole 2ch。数字人声音进直播，本机也能监听。

### 路线 B：RTMP 直推（需满足官方门槛，适合长期无人值守）

官方《推流直播功能指引》的开通条件：**① 完成视频号认证 ② 粉丝 ≥ 500 ③ 公开视频 ≥ 10 ④ 公开直播 ≥ 3 场**。
**不满足时助手后台不显示推流入口**——这就是"找不到创建推流直播"的原因，不是入口藏得深。
达标后：直播管理 → 创建直播 → 获取推流地址 + 密钥 → 填入配置 → `uv run goovo run --sink rtmp`：

```toml
[stream]
sink = "rtmp"
rtmp_url = "rtmp://xxx.livepush.myqcloud.com/live/xxx?txSecret=..."
```

认证路径：手机视频号 → 右上角👤 → 创作者中心 → 创作者服务 → 更多 → 认证。
兴趣认证需有效粉丝 > 1000；不足可走职业 / 企业认证。直播场次用手机先播 3 场即可累计。

### 开播操作注意（来自官方指引）

- 推流地址**生成后 2 小时有效**，关播 / 解散房间即失效；超长直播留意重连时地址是否过期。
- 结束顺序：**先在网页 / 伴侣点"结束直播"，再停本项目**，否则观众看到黑屏。
- 推荐参数已内置：1080×1920@30fps、3000kbps、AAC-LC。
- 断流自动重连；直播中手机打开控制面板（`[panel].host = "0.0.0.0"`）随时插播回复评论。

抖音 / 快手 / B站 / YouTube：按路线 B 方式填入各自 RTMP 地址即可。

**合规提醒**：视频号对录播、虚拟人直播有明确规范（可能要求真人出镜认证、AI 内容标识等）。
请使用**你本人形象与授权声音**，遵守平台规则，必要时在画面/简介标注"AI 辅助直播"。

## 口播内容

- **稿件循环**（默认）：`scripts/*.md`，文件内用单独一行 `---` 分段，每段一次口播。
  支持 `shuffle = true` 随机循环，段间静默 `gap_seconds` 可调。
- **LLM 撰稿**：`[brain].mode = "llm"`，配置人设 `persona`、主题轮播 `topics`，
  设置 `OPENAI_API_KEY`（或改 `api_key_env` 指向你的兼容服务，如 DeepSeek/通义/Kimi 的 OpenAI 兼容端点）。
  LLM 失败时自动回退稿件，直播不断。
- **人工插播**：控制面板或 `goovo say`，优先级最高，可打断当前口播——用来回复评论区提问。

## 声音

**默认用你自己的声音**：从 `talk.mp4` 音轨提取参考音频，F5-TTS 零样本克隆（无需训练）。

```bash
uv venv .venv-clone --python 3.11 && uv pip install --python .venv-clone/bin/python f5-tts
uv run goovo prepare-voice                 # 提取 assets/voice/ref.wav（12s）
.venv-clone/bin/python tools/clone_server.py   # 常驻克隆服务 :9880（首次下载模型约 1.8GB）
uv run goovo warmup                        # 预合成全部口播稿 → 开播即播
uv run goovo run
```

克隆速度约 1.1× 实时（MPS、16 步扩散）；`warmup` 把稿件循环全部缓存后直播零等待，插播/LLM 新稿约等 1 倍时长（面板显示备稿进度）。
音色相似度实测：你的原声 vs 克隆 0.923（说话人嵌入余弦），vs 通用男声 0.536。
克隆服务不可用时自动回退 **edge 男声**（`fallback_voice`），再失败回退 macOS say，直播不断声。

| 引擎 | 配置 | 说明 |
|---|---|---|
| F5-TTS 克隆 | `engine = "http"` | **你的声音**；参考音频 `assets/voice/ref.wav`，服务 `tools/clone_server.py` |
| edge-tts | `engine = "edge"` | 免费微软神经语音，需联网；男声 Yunxi/Yunyang/Yunjian，女声 Xiaoxiao 等 |
| macOS say | `engine = "macos"` | 离线兜底（Tingting），edge 失败时自动回退 |
| 其他克隆服务 | `engine = "http"` | 改 `[voice.http].url/json_body` 指向 GPT-SoVITS / CosyVoice 等 |

合成结果按文本缓存于 `.cache/tts/`，循环稿件第二次起零延迟。
合成音频会自动响度归一（RMS≈0.08、峰值≤0.9），避免克隆服务音量过低导致口型门限不触发。

## 升级路线

神经口型已内置（neural 引擎，Apple GPU）。其他可扩展方向：
评论区抓取→LLM 生成回复→自动插播；多平台同时推流；OBS 虚拟摄像头输出；
更强 GPU（M4/M5）可提高渲染余量或上采样到更高分辨率。

## 常见问题

- **助手后台找不到"推流直播"**：账号未达路线 B 的四个门槛，入口不显示。改用路线 A（Mac 直播伴侣窗口捕获），今天就能播。
- **预览窗口没画面**：确认 `ffplay` 已安装（`brew install ffmpeg` 自带）。
- **卡顿/音画不同步**：降低 `[avatar]` 分辨率到 540×960 或帧率到 20。
- **edge-tts 报错**：网络问题，会自动回退 macOS say；或配置克隆服务。
- **推流失败**：检查 rtmp_url 完整性（含密钥）、视频号助手是否已创建直播、上行带宽 ≥ 4Mbps。
- **停止**：Ctrl+C 或 `kill -TERM <pid>`，录制文件会正常收尾。
- **M2 Pro 的 GPU 用得上吗**：用得上。neural 引擎的 MuseTalk 渲染跑在 Apple GPU（Metal/MLX）上，
  但只在预渲染阶段用；播放阶段纯 CPU（实测 ~12ms/帧）。mouth 兜底引擎全程 CPU，不需要 GPU。

## 目录

```
goovo_human/        核心代码（brain 大脑 / tts 语音 / avatar 形象 / pipeline 合成推流 / panel 控制台）
scripts/            口播稿（--- 分段）
assets/             形象视频（idle.mp4 / talk.mp4）
config.toml         主配置
output/             录制与试听输出
```
# AI-Livestreamer
