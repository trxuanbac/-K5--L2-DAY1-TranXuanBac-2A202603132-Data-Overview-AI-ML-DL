# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**
   - `class_id` = `468`
   - `class_name` = `cab`
   - `rank` = `1`
   - `score` = `0.510915`
   - `taxonomy_name` = `ImageNet-1K`

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**
   - Record này nói về **cả bức ảnh**. Vì mô hình Classification phân loại nội dung tổng thể của ảnh và đưa ra một danh sách các lớp có khả năng phù hợp nhất. Nó không xác định vị trí riêng của từng vật thể.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**
   - Danh sách tên lớp được xác định bởi taxonomy đi kèm checkpoint đã huấn luyện, ở đây là `ImageNet-1K`. Model không tự nghĩ ra tên lớp mới.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**
   - `class_id`: mã số định danh của lớp.
   - `class_name`: tên dễ đọc của lớp.
   - `taxonomy_name`: cho biết lớp thuộc hệ thống phân loại nào.
   - Ghi cả ba giúp tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng mã số hoặc tên lớp khác nhau.

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**
   - Guideline cần quy định tiêu chí xác định chủ đề chính của ảnh, cách ưu tiên khi ảnh có nhiều vật thể và cách chọn một lớp đại diện duy nhất.

6. **Vì sao không được lấy `score` của model làm ground truth?**
   - Score chỉ thể hiện mức độ tin cậy của mô hình đối với dự đoán. Ground truth phải do annotator xác định dựa trên nội dung ảnh và guideline. Vì vậy score cao không đồng nghĩa với nhãn đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record (`sample_id = kitchen`):**
   - `class_name` = `person`
   - `score` = `0.892416`
   - `bbox_xyxy` = `[387.15, 71.12, 498.62, 349.85]`
   - `bbox_width` = `111.47`
   - `bbox_height` = `278.73`

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Bounding box là hình chữ nhật bao quanh vật thể. Tọa độ đầu tiên `(x_min, y_min) = (387.15, 71.12)` xác định góc trên bên trái, còn `(x_max, y_max) = (498.62, 349.85)` xác định góc dưới bên phải. Chiều rộng được tính bằng `x_max - x_min = 111.47` pixel và chiều cao bằng `y_max - y_min = 278.73` pixel. Các giá trị tọa độ được tính theo pixel.

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
   - `threshold = 0.20`: `19` vật thể
   - `threshold = 0.35`: `14` vật thể
   - `threshold = 0.60`: `6` vật thể

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng, mô hình giữ lại nhiều prediction hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Reviewer cần kiểm tra thêm các hộp sai, hộp trùng lặp, vật thể bị nhận diện nhầm và những vật thể không thực sự tồn tại trong ảnh.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**
   - Người gán nhãn cần vẽ một bounding box bao quanh toàn bộ phần vật thể nhìn thấy, không bao gồm quá nhiều vùng nền xung quanh. Box phải có góc trên trái và góc dưới phải chính xác, không bị lệch hoặc cắt mất phần vật thể nhìn thấy.

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - Trên ảnh `kitchen`, có các vật dụng nhà bếp trên bàn/kệ (như chai lọ `bottle`, bát `bowl`, hoặc phần bàn `dining table` ở góc dưới) bị che khuất một phần bởi vật thể đứng trước hoặc bị cắt ở mép ảnh. Tôi sẽ đưa trường hợp này lên Lab Coach để **escalation** nếu guideline chưa quy định rõ cách xử lý. Tôi không tự suy đoán phần vật thể bị che khuất hoặc nằm ngoài ảnh, nhằm đảm bảo tính thống nhất giữa các annotator.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record (`sample_id = kitchen`):**
   - `instance_id` = `kitchen-001`
   - `class_name` = `person`
   - `score` = `0.885132`
   - `polygon_point_count` = `184`
   - `polygon_xy` (5 điểm đầu) =
     ```json
     [
       [432.0, 72.0],
       [430.67, 73.33],
       [426.67, 73.33],
       [425.33, 74.67],
       [422.67, 74.67]
     ]
     ```

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Bounding box chỉ biểu diễn vật thể bằng hình chữ nhật bao quanh, còn polygon gồm nhiều điểm tọa độ bám theo đường biên thực tế của vật thể. Vì vậy polygon thể hiện hình dạng chi tiết hơn và hạn chế phần nền không thuộc vật thể.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` dùng để phân biệt từng đối tượng riêng biệt trong kết quả segmentation. Ví dụ, hai chiếc cốc cùng thuộc class `cup` vẫn có hai `instance_id` khác nhau. Đây không phải class ID vì nó không xác định loại đối tượng, cũng không phải tracking ID vì nó không theo dõi đối tượng qua nhiều khung hình video.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**
   - Người gán nhãn cần vẽ polygon sát theo đường viền của phần vật thể nhìn thấy, sử dụng các điểm tọa độ liên tiếp để mô tả hình dạng. Không vẽ lấn sang vật thể khác hoặc vùng nền không thuộc đối tượng.

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - Trên ảnh `kitchen`, tại khu vực bàn bếp và kệ có nhiều đồ vật như chai lọ (`bottle`), bát (`bowl`), cốc (`cup`) đặt sát chạm vào nhau hoặc người (`person`) đứng che khuất một phần lò nướng (`oven`) phía sau. Nếu ranh giới giữa hai vật thể không rõ ràng hoặc có phần bị che khuất, tôi sẽ đối chiếu guideline trước. Nếu guideline chưa có quy định cụ thể, tôi sẽ **escalation** lên Lab Coach thay vì tự suy đoán đường biên.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class cho toàn bộ ảnh | Ảnh có nhiều vật thể (như ảnh `traffic` có cả xe buýt, ô tô, người đi đường nhưng top-1 chỉ chọn `cab`), dễ chọn sai chủ đề chính | Chọn lớp theo guideline | Kiểm tra lớp có phù hợp nội dung ảnh không |
| Phát hiện vật thể | Một class và một bounding box cho mỗi vật thể | Trên ảnh `kitchen`, các chai lọ/bát cốc nhỏ trên kệ dễ bị bỏ sót hoặc box bị lệch, chồng lấn khi các vật đặt sát nhau | Vẽ box bao quanh vật thể | Kiểm tra class, tọa độ, box trùng và vật thể bị bỏ sót |
| Instance segmentation | Một class và một polygon cho mỗi instance | Trên ảnh `kitchen`, biên các đồ vật nhỏ trên bàn bếp mờ, các vật chạm nhau hoặc người đứng che một phần thiết bị bếp phía sau | Vẽ polygon theo biên từng vật thể | Kiểm tra độ chính xác của mask và ID từng instance |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**
   - Tôi chỉ sử dụng ảnh COCO công khai được cung cấp trong notebook, không tải lên hoặc chia sẻ dữ liệu cá nhân, dữ liệu nội bộ hay thông tin nhạy cảm.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
   - Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi bài thực hành, tôi sẽ dừng xử lý và báo cho Lab Coach hoặc người phụ trách phòng Lab để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
