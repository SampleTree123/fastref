# mixed6_shared_canvas_v3 展示快照

打开 `index.html` 查看。媒体已由服务器符号链接复制成真实文件；网页与 media 必须保持相对目录关系。

包含12个case（8个UE合成样例、4个CoinVE真实样例），当前已有base和step50结果，step256和final仍待填充。详情和来源记录见gallery.json；其中绝对路径仅用于记录服务器来源，不是本地浏览依赖。

共49个MP4、10张PNG；SHA256SUMS记录原始61个文件的SHA-256，可在本目录执行 `sha256sum -c SHA256SUMS` 检查完整性。

若浏览器不允许直接打开本地视频，可在本目录运行 `python -m http.server 8765`，再访问 http://localhost:8765/ 。
