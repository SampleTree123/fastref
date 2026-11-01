# mixed6_shared_canvas_v3 展示快照

打开 `index.html` 查看。媒体为真实文件副本，网页与media必须保持相对目录关系。

包含12个case（8个UE合成样例、4个CoinVE真实样例）。已完成：base、step50、step256；待填充：final。详情和来源见gallery.json；绝对路径仅为服务器来源记录，不是本地浏览依赖。

共61个MP4、10张PNG。SHA256SUMS记录原始73个文件的SHA-256，可执行 `sha256sum -c SHA256SUMS` 检查完整性。

若浏览器不允许直接打开本地视频，可在本目录运行 `python -m http.server 8765`，访问 http://localhost:8765/ 。
