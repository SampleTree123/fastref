# 生成效果固定抽查名单

## 2026-11-08T04:16:36+08:00：14/14轮独立抽帧检查完成

名单固定于2026-11-08T03:48:45+08:00，没有按生成效果换题。按测试前30条链原序，每种编辑取最早两轮；共7类、14轮，恰好集中在前三条链。它用于发现具体问题，不代表全部测试的质量。

这次能确认的现象：场景、发型、服装和取景变化多数在画面中可见；两个动作样例在抽到的时间点呈现参考的大致动作阶段。12419第5轮正确回到第1轮生成视频，恢复旧衣服/旧场景，没有混入后来编辑。

也有明确问题：白发要求有一例未清楚达成；鞋款、鞋色和花纹会偏离参考；近景中的脸/头发糊块会延续到下一轮；换场景后一例人物变得过度黄亮。身份是否准确替换不能靠这组静态帧确认。

14轮实际输入均核对了progress、state路径；其中5轮是root、9轮是生成历史。后9轮的源文件SHA均与其真正父轮输出一致。正确改动与已有瑕疵都可能被后续轮次继承，但这不等于证明所有链都会逐轮变差。

| 链 / 轮 | 编辑 | 计划source | 实际source | 实际输入 |
|---|---|---:|---:|---|
| test_chain_13332 / 1 | 发色 | 0 | 0 | 原始视频 |
| test_chain_13332 / 2 | 场景 | 0 | 0 | 原始视频 |
| test_chain_13332 / 3 | 发型 | 2 | 2 | 真实生成历史 |
| test_chain_13332 / 4 | 服装 | 3 | 3 | 真实生成历史 |
| test_chain_13332 / 5 | 动作 | 4 | 4 | 真实生成历史 |
| test_chain_12727 / 1 | 发色 | 0 | 0 | 原始视频 |
| test_chain_12727 / 2 | 发型 | 0 | 0 | 原始视频 |
| test_chain_12727 / 3 | 镜头 | 2 | 2 | 真实生成历史 |
| test_chain_12727 / 4 | 身份 | 3 | 3 | 真实生成历史 |
| test_chain_12419 / 1 | 身份 | 0 | 0 | 原始视频 |
| test_chain_12419 / 2 | 服装 | 1 | 1 | 真实生成历史 |
| test_chain_12419 / 3 | 动作 | 2 | 2 | 真实生成历史 |
| test_chain_12419 / 4 | 场景 | 3 | 3 | 真实生成历史 |
| test_chain_12419 / 5 | 镜头 | 1 | 1 | 真实生成历史 |

计划source只是原数据路线；真实输入从运行记录读取。GT仅作离线属性/照明对照，不进入模型或替换历史。正常样例看1/4/8秒；两个动作样例看1/2/4/6/8秒的实际源、输出、参考和GT；最后镜头样例增加0/9秒。输出比原始素材分辨率低，不能把全部细节损失归于编辑。

### test_chain_13332_turn_01

要求：把红色头发变成白色，保留其余内容。

看到的变化：1/4秒头发主要是深灰或黑灰；8秒仍有明显红色部分。白发要求未清楚达成。离线补看GT同时间帧：GT三帧均为明亮白发，8秒虽有暖光但没有输出那种大面积深灰/红色，因此当前差异不能仅由同场景照明解释。

保留内容：黑白服装、蓝鞋、屋顶环境和三个时点的手臂/身体姿态大体保留。

问题与未确认部分：头发改色不完整；塔架与楼房边缘有扭曲和细节损失。输出比高分辨率源更软，不能把全部清晰度差异都归于编辑错误。 只判断这三个时点；不能由此判断全程是否闪烁或动作是否流畅。

实际源：[1秒](generated_quality_frames/c50f0c4d212f_1s.png)、[4秒](generated_quality_frames/c50f0c4d212f_4s.png)、[8秒](generated_quality_frames/c50f0c4d212f_8s.png)；实际输出：[1秒](generated_quality_frames/71bdb77c1b73_1s.png)、[4秒](generated_quality_frames/71bdb77c1b73_4s.png)、[8秒](generated_quality_frames/71bdb77c1b73_8s.png)。

### test_chain_13332_turn_02

要求：换成参考中的街道环境，只允许环境及其照明影响改变，保留人物和镜头/动作。

看到的变化：屋顶变成旧式石砌建筑与铺石街道，与场景参考的内容类别相符；人物受光和投影也变化。

保留内容：原红发、黑白服装、蓝鞋和抬手/叉腰姿态大体保留。三个时点人物在画面中的位置和拍摄侧面关系大体对应。

问题与未确认部分：背景窗框、建筑线条局部弯曲和模糊；8秒腰部/白布轮廓与源有小差异。没有据静态帧给相机运动打分。 参考只有一张初始画面，不能用它逐像素判断后续视角对错。

实际源：[1秒](generated_quality_frames/c50f0c4d212f_1s.png)、[4秒](generated_quality_frames/c50f0c4d212f_4s.png)、[8秒](generated_quality_frames/c50f0c4d212f_8s.png)；实际输出：[1秒](generated_quality_frames/95537387cfc4_1s.png)、[4秒](generated_quality_frames/95537387cfc4_4s.png)、[8秒](generated_quality_frames/95537387cfc4_8s.png)。

### test_chain_13332_turn_03

要求：在轮2生成的街道视频上，参考正背图换发型，只转移发型形状。

看到的变化：后脑低扎发改成头顶双丸子/发髻轮廓，与参考和GT的主要结构相符，红色头发保留。

保留内容：轮2生成的街道布局、黑白衣服、蓝鞋和抬手/叉腰姿态继续保留，没有回到原始屋顶。

问题与未确认部分：发髻大小和发丝细节不完全清楚；背景沿用上一轮的弯曲线条/模糊。不能把既有背景问题都归于本轮。 已核对实际source是轮2生成文件，属于真实历史回填。GT街道视角与生成源不同，只对照目标发型，不用GT背景要求生成结果完全一致。

实际源：[1秒](generated_quality_frames/95537387cfc4_1s.png)、[4秒](generated_quality_frames/95537387cfc4_4s.png)、[8秒](generated_quality_frames/95537387cfc4_8s.png)；实际输出：[1秒](generated_quality_frames/9fe88b59af9a_1s.png)、[4秒](generated_quality_frames/9fe88b59af9a_4s.png)、[8秒](generated_quality_frames/9fe88b59af9a_8s.png)；离线GT：[1秒](generated_quality_frames/01cd666e53f9_1s.png)、[8秒](generated_quality_frames/01cd666e53f9_8s.png)。

### test_chain_13332_turn_04

要求：在轮3结果上把黑白衣服和蓝鞋替换为参考的整套花裙与鞋。

看到的变化：衣服明显变成白底多色花裙，鞋也改变；但输出鞋偏鲜红，参考为棕粉色、GT为浅粉色，颜色不完全一致。花纹比参考/GT更浓艳、更粗。

保留内容：轮3双丸子头、红发、生成街道和三个时点的身体姿态大体保留。

问题与未确认部分：花纹和衣物边缘有模糊，鞋颜色偏差明显；局部身体亮度有所变化。 有GT和参考的离线对照；没有将正常换装轮廓改变都算作保留失败。静态帧无法评价整段花纹闪烁。

实际源：[1秒](generated_quality_frames/9fe88b59af9a_1s.png)、[4秒](generated_quality_frames/9fe88b59af9a_4s.png)、[8秒](generated_quality_frames/9fe88b59af9a_8s.png)；实际输出：[1秒](generated_quality_frames/f3895a8a517e_1s.png)、[4秒](generated_quality_frames/f3895a8a517e_4s.png)、[8秒](generated_quality_frames/f3895a8a517e_8s.png)；离线GT：[1秒](generated_quality_frames/b26299c14e16_1s.png)、[8秒](generated_quality_frames/b26299c14e16_8s.png)。

### test_chain_13332_turn_05

要求：沿用轮4生成的花裙视频，只把动作替换为参考视频所示动作。

看到的变化：1/2秒站立，4/6/8秒出现迈步和手臂摆动；在这些时点上与参考、GT的大致动作阶段相符，替换了原来的抬手/叉腰动作。

保留内容：生成街道、红发双丸子头、花裙和此前偏红的鞋继续保留，没有复制动作参考的青色运动服或灰色背景。

问题与未确认部分：花纹和背景已有的模糊/弯曲继续存在；侧身处裙摆与身体轮廓有简化，前脸细节难与之前背面源精确比较。 查看1/2/4/6/8秒实际源、输出、参考和GT；只能说明选中时点有动作阶段对应，不能证明整段时序、步速、足部接触或连续运动正确。

实际源：[1秒](generated_quality_frames/f3895a8a517e_1s.png)、[2秒](generated_quality_frames/f3895a8a517e_2s.png)、[4秒](generated_quality_frames/f3895a8a517e_4s.png)、[6秒](generated_quality_frames/f3895a8a517e_6s.png)、[8秒](generated_quality_frames/f3895a8a517e_8s.png)；实际输出：[1秒](generated_quality_frames/d682a557c986_1s.png)、[2秒](generated_quality_frames/d682a557c986_2s.png)、[4秒](generated_quality_frames/d682a557c986_4s.png)、[6秒](generated_quality_frames/d682a557c986_6s.png)、[8秒](generated_quality_frames/d682a557c986_8s.png)；离线GT：[1秒](generated_quality_frames/5fa8f83b9dd6_1s.png)、[2秒](generated_quality_frames/5fa8f83b9dd6_2s.png)、[4秒](generated_quality_frames/5fa8f83b9dd6_4s.png)、[6秒](generated_quality_frames/5fa8f83b9dd6_6s.png)、[8秒](generated_quality_frames/5fa8f83b9dd6_8s.png)；动作参考：[1秒](generated_quality_frames/555911305fc3_1s.png)、[2秒](generated_quality_frames/555911305fc3_2s.png)、[4秒](generated_quality_frames/555911305fc3_4s.png)、[6秒](generated_quality_frames/555911305fc3_6s.png)、[8秒](generated_quality_frames/555911305fc3_8s.png)。

### test_chain_12727_turn_01

要求：把白发变成金发，保留原发型及其他内容。

看到的变化：三个时点都从白发变成明显黄/金色；色调较鲜亮。

保留内容：粉色上衣、白裙、黑靴、水边遗迹和姿态大体保留，原盘/扎起的发型轮廓仍可见。

问题与未确认部分：头发和衣服细节变软；人物远且头部很小，无法判断发丝细节是否稳定。 这表示颜色修改可见，不等于自然度或整体质量的量化结论。

实际源：[1秒](generated_quality_frames/c67a7163975b_1s.png)、[4秒](generated_quality_frames/c67a7163975b_4s.png)、[8秒](generated_quality_frames/c67a7163975b_8s.png)；实际输出：[1秒](generated_quality_frames/b36d67343b61_1s.png)、[4秒](generated_quality_frames/b36d67343b61_4s.png)、[8秒](generated_quality_frames/b36d67343b61_8s.png)。

### test_chain_12727_turn_02

要求：参考正背图换发型，只移植发型形状；保留原人物白发及其他内容。

看到的变化：后脑轮廓变成较短的披散/蓬起形状，原扎发轮廓减弱，白发保留。

保留内容：没有复制参考的黑发；衣服、背景和三个时点姿态大体保留。

问题与未确认部分：头部像素太少，不能确认是否复现了参考的分缝、卷曲和后脑结构；不把轮廓变化直接计作完全成功。 与参考两张图对照过，但不能判断人脸身份是否精确保留。

实际源：[1秒](generated_quality_frames/c67a7163975b_1s.png)、[4秒](generated_quality_frames/c67a7163975b_4s.png)、[8秒](generated_quality_frames/c67a7163975b_8s.png)；实际输出：[1秒](generated_quality_frames/660597f903cb_1s.png)、[4秒](generated_quality_frames/660597f903cb_4s.png)、[8秒](generated_quality_frames/660597f903cb_8s.png)。

### test_chain_12727_turn_03

要求：在轮2生成视频上改为正面脸部近景，用镜头参考确定初始构图，保持动作时序和其余内容。

看到的变化：侧面全身明确变成正面脸部近景，初始构图接近参考；白发、粉色衣服和遗迹场景大体保留。

保留内容：实际输入是轮2生成结果；输出保留白色短发而没有继承被回溯舍弃的金发。

问题与未确认部分：4秒脸部/手部明显变糊、头发呈块状；8秒头发和低头轮廓失真。4秒输出的手/头姿态与同时间GT不同，未能确认动作时序按要求保留。GT同时间帧相对清楚，不能仅以同场景照明解释这些差异。 镜头变化本身可见；不能从三帧量化整段运动质量或精确身份一致性。镜头参考含目标人物画面，不能把初始脸部接近参考单独归因于源视频信息保持。

实际源：[1秒](generated_quality_frames/660597f903cb_1s.png)、[4秒](generated_quality_frames/660597f903cb_4s.png)、[8秒](generated_quality_frames/660597f903cb_8s.png)；实际输出：[1秒](generated_quality_frames/9bd36d72295e_1s.png)、[4秒](generated_quality_frames/9bd36d72295e_4s.png)、[8秒](generated_quality_frames/9bd36d72295e_8s.png)；离线GT：[1秒](generated_quality_frames/924c646c646a_1s.png)、[4秒](generated_quality_frames/924c646c646a_4s.png)、[8秒](generated_quality_frames/924c646c646a_8s.png)。

### test_chain_12727_turn_04

要求：在轮3近景上按照片替换身份与肤色，保留发型、发色、服装和镜头等。

看到的变化：脸部和胸颈肤色明显加深；但比参考照片和同场景GT更暗，高光很强，面部结构是否准确匹配参考尚不清楚。

保留内容：正面近景、白发、粉衣和生成背景继续保留。轮3在4/8秒已有的头发糊块/姿态形变，在本轮相同时间位置继续出现。

问题与未确认部分：深色皮肤上有突兀的亮斑，4/8秒脸部暗区和模糊更难分辨；不能把“变黑”直接视为身份替换成功。 参考/GT只作离线比对；两者光照并不完全一致，不能精确比较肤色数值，但GT同场景下没有输出这种明显加深程度。无法用三帧断言身份识别成功或给总体得分。

实际源：[1秒](generated_quality_frames/9bd36d72295e_1s.png)、[4秒](generated_quality_frames/9bd36d72295e_4s.png)、[8秒](generated_quality_frames/9bd36d72295e_8s.png)；实际输出：[1秒](generated_quality_frames/7a54cfe808a5_1s.png)、[4秒](generated_quality_frames/7a54cfe808a5_4s.png)、[8秒](generated_quality_frames/7a54cfe808a5_8s.png)；离线GT：[1秒](generated_quality_frames/2d84ae616613_1s.png)、[4秒](generated_quality_frames/2d84ae616613_4s.png)、[8秒](generated_quality_frames/2d84ae616613_8s.png)。

### test_chain_12419_turn_01

要求：按参考照片替换人物身份，身体肤色与目标一致，保留发型等其余内容。

看到的变化：脸部轮廓有小变化，但实际源与GT的身份外观本就接近，且红色场景光很强，不能据这几个帧确认已经准确换成参考身份。

保留内容：原针织衫、黑裤、靴子、发型和霓虹穹顶场景保留。

问题与未确认部分：4秒脸、手臂和衣服呈糊块，8秒针织纹理变粗和变形。红色皮肤在原视频和GT也出现，不能直接当作身份编辑导致的肤色错误。 对照参考照片及GT1/8秒，但不做身份准确率或肤色数值判断。

实际源：[1秒](generated_quality_frames/fbf5c0a72bd3_1s.png)、[4秒](generated_quality_frames/fbf5c0a72bd3_4s.png)、[8秒](generated_quality_frames/fbf5c0a72bd3_8s.png)；实际输出：[1秒](generated_quality_frames/980b4f96fa3a_1s.png)、[4秒](generated_quality_frames/980b4f96fa3a_4s.png)、[8秒](generated_quality_frames/980b4f96fa3a_8s.png)；离线GT：[1秒](generated_quality_frames/0f24562be0ab_1s.png)、[8秒](generated_quality_frames/0f24562be0ab_8s.png)。

### test_chain_12419_turn_02

要求：在轮1生成结果上整套换成参考西装和鞋。

看到的变化：针织衫/黑裤变为格纹西装、马甲、白衬衫、领带和同款裤子，服装类别明显完成变化。

保留内容：原发型、霓虹背景、身体动作阶段继续保留。西装偏红在同场景GT也存在，与红色灯光有关。

问题与未确认部分：输出鞋为带明显浅色条带/鞋头区域的鞋，和参考及GT黑色正装鞋不同；扣子和格纹较粗，4秒沿用前轮脸/手模糊。 GT和参考用于区分灯光导致的颜色变化与实际鞋款不符，没有要求输出逐像素等于GT。

实际源：[1秒](generated_quality_frames/980b4f96fa3a_1s.png)、[4秒](generated_quality_frames/980b4f96fa3a_4s.png)、[8秒](generated_quality_frames/980b4f96fa3a_8s.png)；实际输出：[1秒](generated_quality_frames/e9a2de3b0221_1s.png)、[4秒](generated_quality_frames/e9a2de3b0221_4s.png)、[8秒](generated_quality_frames/e9a2de3b0221_8s.png)；离线GT：[1秒](generated_quality_frames/0ecab19c1e24_1s.png)、[8秒](generated_quality_frames/0ecab19c1e24_8s.png)。

### test_chain_12419_turn_03

要求：沿用轮2西装结果，替换为参考中的行走动作。

看到的变化：前段站立，随后出现向镜头前进的身体/手臂姿态，8秒抬头；1/2/4/6/8秒的大致阶段与参考和GT相符。

保留内容：西装、发型、霓虹场景保留；没有复制参考的运动服和灰色背景。

问题与未确认部分：前轮错误鞋款和粗糙格纹延续；4秒脸部模糊，手部细节不清。 所查帧支持动作发生改变及大致阶段对应，不能证明连续步态、精确速度、每一帧的相位正确。

实际源：[1秒](generated_quality_frames/e9a2de3b0221_1s.png)、[2秒](generated_quality_frames/e9a2de3b0221_2s.png)、[4秒](generated_quality_frames/e9a2de3b0221_4s.png)、[6秒](generated_quality_frames/e9a2de3b0221_6s.png)、[8秒](generated_quality_frames/e9a2de3b0221_8s.png)；实际输出：[1秒](generated_quality_frames/8d0340f296cc_1s.png)、[2秒](generated_quality_frames/8d0340f296cc_2s.png)、[4秒](generated_quality_frames/8d0340f296cc_4s.png)、[6秒](generated_quality_frames/8d0340f296cc_6s.png)、[8秒](generated_quality_frames/8d0340f296cc_8s.png)；离线GT：[1秒](generated_quality_frames/dfcade8d2f72_1s.png)、[2秒](generated_quality_frames/dfcade8d2f72_2s.png)、[4秒](generated_quality_frames/dfcade8d2f72_4s.png)、[6秒](generated_quality_frames/dfcade8d2f72_6s.png)、[8秒](generated_quality_frames/dfcade8d2f72_8s.png)；动作参考：[1秒](generated_quality_frames/47ca104209c0_1s.png)、[2秒](generated_quality_frames/47ca104209c0_2s.png)、[4秒](generated_quality_frames/47ca104209c0_4s.png)、[6秒](generated_quality_frames/47ca104209c0_6s.png)、[8秒](generated_quality_frames/47ca104209c0_8s.png)。

### test_chain_12419_turn_04

要求：在轮3行走视频上换成参考的桥下道路场景，允许适应环境照明。

看到的变化：霓虹穹顶替换为桥梁、道路、秋树和施工标识；环境类别与参考相符，行走/抬头姿态仍可见。

保留内容：格纹西装、黑领带和发型保留；前轮带浅色条带的鞋继续出现。

问题与未确认部分：人物和衣服整体呈强烈黄/金色，脸部非常光滑；GT同场景中人物主要处于阴影、肤色与西装较自然。差异不能只用“换了场景所以换光”概括，但不做跨图精确色度判断。 场景修改允许照明改变，仍需分别看环境达成与人物外观失真；三帧不能判断全段光照是否稳定。

实际源：[1秒](generated_quality_frames/8d0340f296cc_1s.png)、[4秒](generated_quality_frames/8d0340f296cc_4s.png)、[8秒](generated_quality_frames/8d0340f296cc_8s.png)；实际输出：[1秒](generated_quality_frames/7e4a3fe95a26_1s.png)、[4秒](generated_quality_frames/7e4a3fe95a26_4s.png)、[8秒](generated_quality_frames/7e4a3fe95a26_8s.png)；离线GT：[1秒](generated_quality_frames/0a3f5e680fb2_1s.png)、[8秒](generated_quality_frames/0a3f5e680fb2_8s.png)。

### test_chain_12419_turn_05

要求：明确回到第1轮结果，改为从上半身拉远到全身的正面镜头。

看到的变化：0/1秒上半身较近，8/9秒全身较远；取景变化方向与要求、参考和GT相符。

保留内容：实际source为轮1生成文件而非轮4，输出恢复针织衫和霓虹背景，没有错误继承后来西装、桥下场景或行走编辑。

问题与未确认部分：4秒动作处脸和衣服仍明显模糊；针织纹理粗化和扭曲沿用了早期生成结果的问题。 查看0/1/4/8/9秒；这些帧支持回溯选择和取景尺度方向正确，不能完整判断相机轨迹平滑性或整段动作时序。

实际源：[0秒](generated_quality_frames/980b4f96fa3a_0s.png)、[1秒](generated_quality_frames/980b4f96fa3a_1s.png)、[4秒](generated_quality_frames/980b4f96fa3a_4s.png)、[8秒](generated_quality_frames/980b4f96fa3a_8s.png)、[9秒](generated_quality_frames/980b4f96fa3a_9s.png)；实际输出：[0秒](generated_quality_frames/188315fe63c5_0s.png)、[1秒](generated_quality_frames/188315fe63c5_1s.png)、[4秒](generated_quality_frames/188315fe63c5_4s.png)、[8秒](generated_quality_frames/188315fe63c5_8s.png)、[9秒](generated_quality_frames/188315fe63c5_9s.png)；离线GT：[1秒](generated_quality_frames/e4f5ccaadba5_1s.png)、[8秒](generated_quality_frames/e4f5ccaadba5_8s.png)。

### 检查范围与不能推导的结论

14/14表示14轮均完成上述抽帧检查，不是看完14段完整视频。没有给整体质量分数，没有完整评价闪烁、动作流畅性、相机轨迹或身份准确率，也没有把原始计划选源一致当作生成质量。镜头参考本身含目标人物画面，画面接近参考不能单独证明源视频所有信息都保留。

原始固定名单只涉及3条链，观察到的成功或失败不能当作整个测试集的比例。GT用来帮助区分照明和外观变化，不要求生成历史逐像素与原图路线一致。

13332第1轮补看GT照明：[1秒](generated_quality_frames/13332_turn01_gt_1s.png)、[4秒](generated_quality_frames/13332_turn01_gt_4s.png)、[8秒](generated_quality_frames/13332_turn01_gt_8s.png)。

[完整名单、实际路径和SHA证据](generated_quality_sample_plan.json) · [抽帧对应关系](generated_quality_frame_evidence.json) · [首批生成历史路径审查](first_formal_generated_history.json)
