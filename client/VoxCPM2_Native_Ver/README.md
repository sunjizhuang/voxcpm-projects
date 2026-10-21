
License：本客户端仅作为 VoxCPM2 server 的调用工具，遵循 VoxCPM 主项目的许可协议。
VoxCPM2 客户端依赖
说明：
- 本客户端可在无第三方依赖下运行（HTTP/SSE 后端）。
- 强烈推荐安装 gradio_client，可自动处理文件上传、SSE 协议，
并兼容不同 Gradio 版本。
安装：
pip install -r requirements.txt
gradio_client>=1.3.0


1. 启动 server
bash
python app_ori.py --host 127.0.0.1 --port 8808
2. 运行客户端
一次性命令（示例：声音设计）：
python voxcpm_client.py --text "你好，世界" --control "年轻女性，温柔甜美"
python voxcpm_client.py --text "你好世界，这是参考音频的文字内容 ，请你在回答前先向我提问， 要求只问一个问题，请根据我的回答继续追问，直到你完全理解我的真实需求和目标时" --control "年轻女性，温柔甜美" --max-chars 18 --no-play 

交互式 REPL：
python voxcpm_client.py
启动后即进入 voxcpm> 提示符，输入命令进行交互。

一次性命令用法
只要在命令行中提供了任意业务参数（--text、--ref、--control 等），客户端就会以「一次性命令」模式运行，生成完成后退出。

示例 1：Voice Design（仅描述，无参考音频）
python voxcpm_client.py --text "你好，世界"  --control "年轻女性，温柔甜美"
  
示例 2：Controllable Cloning（参考音频 + 风格控制）
python voxcpm_client.py --ref /path/to/female_voice.wav --control "语速稍快，开心" --text "今天天气不错"

示例 3：Ultimate Cloning（参考音频 + transcript）
python voxcpm_client.py \
  --ultimate \
  --ref /path/to/reference.wav \
  --transcript "这是参考音频的文字内容" \
  --text "接下来继续说的话"
  
示例 4：调整高级参数
python voxcpm_client.py \
  --text "测试文本" \
  --control "中年男性，低沉" \
  --cfg 2.5 \
  --steps 16 \
  --normalize \
  --seed 12345
  
示例 5：连接远程 server
python voxcpm_client.py \
  --server http://192.168.1.10:8808 \
  --text "远程生成测试"
  
示例 6：指定输出目录，关闭自动播放
python voxcpm_client.py \
  --text "..." \
  --out-dir ./my_audio \
  --no-play

  
交互式 REPL 用法
不带任何业务参数运行：
python voxcpm_client.py

进入 REPL 后：

text
===== VoxCPM2 交互式客户端 =====
server = http://127.0.0.1:8808  api = /generate
输入 :h 查看帮助，:q 退出。

voxcpm>
典型会话
text
voxcpm> :mode
请选择生成模式：
  1) Voice Design         （仅 Control Instruction）
  2) Controllable Cloning （参考音频 + 可选 Control Instruction）
  3) Ultimate Cloning     （参考音频 + 手动输入 transcript）
输入 1/2/3 [1]: 2
参考音频路径: /home/user/ref.wav
Control Instruction（可选）: 语速稍快，开心
Target Text: 今天天气不错
是否调整高级参数 (y/n) [n]: n
voxcpm> :run

▶ 提交生成：mode=Controllable Cloning, cfg=2.0, steps=10, ...
✅ 生成成功
   后端    : gradio_client
   音频    : /path/outputs/voxcpm_20250101_121500.wav
   采样率  : 24000
   实际seed: 1842166930
   耗时    : 5.1s

voxcpm> :q
再见 👋
REPL 命令速查
命令	说明
:h / :help / ?	显示帮助
:mode	交互式选择模式并逐项填写参数
:ref <path>	设置参考音频路径（留空则清除）
:text <string>	设置目标文本
:ctrl <string>	设置 Control Instruction
:trans <string>	设置参考音频 transcript（极致克隆用）
:cfg <float>	设置 CFG 引导强度（1.0–3.0）
:steps <int>	设置 LocDiT 步数
:norm on|off	文本规范化开关
:denoise on|off	参考音频降噪开关
:seed <int|rand>	设置固定 seed 或恢复随机 seed
:run	使用当前参数生成
:show	显示当前所有参数
:server <url>	临时切换 server（自动重建后端）
:q / :quit / :exit	退出
命令行参数
参数	默认值	说明
--server	http://127.0.0.1:8808	server 地址（也可用环境变量 VOXCPM_SERVER）
--port	无	覆盖 server 端口
--api-name	/generate	Gradio API 名称
--backend	auto	通信后端：auto / gradio / http
--text	None	目标文本
--control	空	Control Instruction
--ref	无	参考音频路径
--transcript	空	参考音频 transcript（极致克隆用）
--ultimate	False	开启极致克隆模式
--cfg	2.0	CFG 引导强度
--steps	10	LocDiT 步数
--normalize	False	开启文本规范化
--denoise	False	开启参考音频降噪
--seed	None	固定 seed
--random-seed	False	每次请求前随机 seed（与 --seed 互斥）
--out-dir	./outputs	音频输出目录
--no-play	False	不自动播放生成的音频
三种生成模式
1. Voice Design（声音设计）
所需输入：--text + --control

不需要参考音频

模型根据 Control Instruction 从零创造音色

bash
python voxcpm_client.py \
  --text "你好，世界" \
  --control "深宫太后，声音低沉阴冷，语速缓慢而有力"
2. Controllable Cloning（可控克隆）
所需输入：--ref + --text，--control 可选

保留参考音频音色，可用 Control Instruction 控制情绪、语速、风格

bash
python voxcpm_client.py \
  --ref female.wav \
  --control "语速稍快，开心" \
  --text "今天天气不错"
3. Ultimate Cloning（极致克隆）
所需输入：--ultimate + --ref + --transcript + --text

模型将参考音频视为「已说出的前文」，以音频续写方式还原全部声音细节

注意：此模式会忽略 Control Instruction

bash
python voxcpm_client.py \
  --ultimate \
  --ref reference.wav \
  --transcript "大家好，我是小美，欢迎收听今天的节目。" \
  --text "今天我们要聊的话题是人工智能。"
💡 transcript 从哪来？ server 端的 ASR 没有独立暴露 api，client 侧不会自动识别。你可以：

使用 VoxCPM server 界面手动跑一次 ASR，把结果拷过来

或本地用 funasr / whisper 等工具识别后填入

通信后端
客户端支持两种后端，可通过 --backend 控制：

后端	说明	依赖
gradio	官方 gradio_client，自动处理文件上传、SSE 协议	pip install gradio_client
http	原生 HTTP + SSE 调用 /gradio_api/call/generate	无（标准库）
auto（默认）	优先 gradio_client，导入失败自动回退 http	—
为什么推荐 gradio_client？
自动把参考音频上传到 Gradio 缓存目录（否则 server 会拒绝本地路径）

官方维护，兼容 Gradio 版本变化

SSE 返回格式解析更稳定

HTTP/SSE 后端如何处理文件上传？
不依赖 gradio_client 时，客户端会自动：

调用 POST {server}/gradio_api/upload 上传参考音频

拿到 Gradio 缓存中的服务端路径

用该路径构造 FileData 并提交

这样可以避免报错：

text
Cannot move ... to the gradio cache dir because it was not uploaded by a user.
输出与播放
保存位置
生成结果统一保存到 --out-dir 指定的目录（默认 ./outputs/），文件名格式：

text
voxcpm_YYYYmmdd_HHMMSS.wav
打印信息
每次生成后，客户端会打印：

text
✅ 生成成功
   后端    : gradio_client
   音频    : /path/outputs/voxcpm_20250101_121500.wav
   采样率  : 24000
   实际seed: 1842166930
   耗时    : 5.1s
其中 实际seed 是本次生成真正使用的 seed，可用于复现。

自动播放
检测到以下任一播放器时，会自动播放一次：

ffplay（FFmpeg）

afplay（macOS）

aplay / paplay（Linux）

关闭自动播放：

bash
python voxcpm_client.py --no-play ...
错误处理
场景	一次性命令	REPL 模式
server 未启动 / 不通	打印错误，exit(1)	打印错误，继续下一轮
参考音频不存在	打印错误，exit(1)	打印错误，继续下一轮
参数不合法（如极致克隆缺 transcript）	打印错误，exit(1)	打印错误，继续下一轮
server 返回生成失败	打印错误，exit(1)	打印错误，继续下一轮
自动播放失败	打印警告，不影响退出码	打印警告，继续下一轮
常见问题
Q1：ModuleNotFoundError: No module named 'gradio_client'
现象：

text
⚠️  gradio_client 不可用（...），回退到 HTTP/SSE。
处理：这是正常提示，客户端已自动回退到 HTTP/SSE 后端。如果想用官方 client：

bash
pip install gradio_client
Q2：The 'meta' field must be explicitly provided ...
原因：参考音频字段格式不对（旧版 Gradio 接受裸路径，新版需要 FileData）。

处理：使用本仓库最新版 voxcpm_client.py 即可（已修复）。

Q3：Cannot move ... to the gradio cache dir because it was not uploaded by a user.
原因：Gradio 出于安全考虑，只接受已上传到缓存目录的文件。

处理：

推荐：pip install gradio_client

或：使用本仓库最新版，HTTP/SSE 后端会自动上传参考音频

Q4：AttributeError: 'list' object has no attribute 'get'
原因：Gradio 版本返回的 SSE complete 事件是数组而非 {"data": [...]}。

处理：使用本仓库最新版（已兼容两种格式）。

Q5：生成太慢 / 卡住
检查 server 端日志是否在运行

增大 --steps 会显著变慢，尝试用默认 10

GPU 显存不足时会自动降速或失败，检查 server 日志

Q6：ffplay: command not found
Ubuntu：sudo apt install ffmpeg

macOS：brew install ffmpeg

或忽略警告，用 --no-play 关闭自动播放

Q7：如何复现某次生成？
使用打印出的「实际seed」：

bash
python voxcpm_client.py --seed 1842166930 ...
REPL 下：

text
voxcpm> :seed 1842166930
voxcpm> :run
Q8：我想在 REPL 里临时换 server
text
voxcpm> :server http://192.168.1.10:8808
  ✅ 已切换 server → http://192.168.1.10:8808（backend=http_sse）
关键设计说明
关注点	处理方式
输入顺序	GENERATE_INPUT_ORDER 严格对应 server 的 inputs=[...]：text, control, ref, use_prompt_text, prompt_text, cfg, normalize, denoise, dit_steps, seed
模式互斥	极致克隆开启时自动清空 control instruction，与 server 端逻辑一致
seed 管理	随机 seed 在客户端生成并回填打印；server 返回真实 seed 时再次回填
FileData	_to_filedata() 把本地路径包装成 Gradio 新版 FileData dict
文件上传	HTTP/SSE 后端 _prepare_payload() 自动上传 FileData 中的本地路径
SSE 兼容	_stream_result() 同时处理 {"data": [...]} 与 [...] 两种返回格式
后端回退	_build_backend() 按 --backend 决定；auto 优先 gradio_client
