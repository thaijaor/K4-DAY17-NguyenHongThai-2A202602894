# Bước 8 — Phân tích kết quả

Số liệu từ `python src/benchmark.py` (offline, deterministic).

| Suite | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|---|
| Standard | Baseline | 2418 | 15065 | 0.00 | 0.00 | 0 | 0 |
| Standard | Advanced | 2742 | 26063 | 1.00 | 1.00 | 305 | 0 |
| Stress | Baseline | 2588 | 22170 | 0.00 | 0.00 | 0 | 0 |
| Stress | Advanced | 3089 | 11045 | 1.00 | 1.00 | 241 | 4 |

## 1. Vì sao Advanced có recall tốt hơn Baseline

Baseline chỉ giữ message theo `thread_id`. Câu hỏi recall được hỏi ở thread mới nên Baseline không có dữ liệu nào để trả lời, recall bằng 0. Advanced rút fact ổn định (tên, nơi ở, nghề, style, đồ uống, món ăn, thú cưng) vào `User.md` ngay khi người dùng nói, rồi đọc lại file này ở mọi thread. Vì `upsert_fact` ghi đè, Advanced trả lời đúng bản đính chính mới nhất: Huế và MLOps engineer ở bộ Standard, Đà Nẵng ở bộ Stress. Các thông tin nhiễu như Hà Nội (chỉ đi họp) hay product manager (câu đùa) không được ghi vào.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

Mỗi thread của bộ Standard chỉ khoảng 450 token, chưa chạm ngưỡng compact 800 token, nên compact không mang lại gì. Trong khi đó, mỗi lượt Advanced vẫn phải gửi thêm `User.md` (khoảng 80 token) và system prompt dài hơn, cộng thêm token để ghi memory. Kết quả là prompt tokens tăng 73% và agent tokens tăng 13%. Memory dài hạn có chi phí cố định ở mọi lượt, và chi phí này chỉ được bù lại khi hội thoại đủ dài.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài

Baseline gửi lại toàn bộ thread ở mỗi lượt: lượt thứ k mang theo k message, nên tổng prompt tăng theo bậc hai. Advanced compact 4 lần trong 16 lượt. Mỗi lần, message cũ được thay bằng một summary có giới hạn độ dài, chỉ giữ lại 4 message gần nhất. Nhờ vậy ngữ cảnh mỗi lượt gần như không đổi (khoảng ngưỡng + `User.md` + summary), và prompt tokens giảm 50%. Compact chủ yếu tối ưu `prompt tokens processed`; còn `agent tokens only` vẫn tăng 19% vì phải sinh summary và ghi memory. Fact quan trọng đã được rút vào `User.md` trước khi compact, nên summary có mất chi tiết thì recall vẫn giữ được 1.00.

Chạy live với `gemini-3.5-flash-lite` cho cùng xu hướng: bộ Standard tốn hơn 157% prompt tokens, bộ Stress giảm 52%.

## 4. File memory tăng trưởng ra sao, rủi ro gì

`User.md` tăng theo số loại fact chứ không theo số lượt: 305 byte sau 101 lượt, vì mỗi fact được ghi đè và `interests` giới hạn tối đa 6 mục. Rủi ro:

- **Lưu sai fact:** fact sai sẽ sai ở mọi phiên sau. Khi chạy live, LLM từng tự ghi chủ đề tin tức ("Energy policy") và chú thích tự bịa vào `User.md`.
- **File phình to:** nếu schema mở thì file phình dần, và vì `User.md` được gửi ở mọi lượt nên càng to càng tốn token.
- **Fact cũ bị giữ lại:** sở thích hay nơi ở cũ còn nằm trong file nếu không có correction hoặc decay.
- **Summary mất chi tiết:** summary dựa trên heuristic có thể bỏ sót những chi tiết chưa kịp đưa vào `User.md`.

## Bonus

| Bonus | Cách làm | Lợi ích | Rủi ro |
|---|---|---|---|
| Confidence threshold | Mỗi fact có confidence; mệnh đề giả định (`nếu`) chỉ 0.4, dưới ngưỡng 0.6 thì không ghi; câu hỏi và yêu cầu nhắc lại bị bỏ qua | Không lưu "Nếu sau này mình nhắc Đà Nẵng..." hay "đồ uống yêu thích là gì" | Ngưỡng cứng có thể bỏ sót fact thật được nói theo kiểu giả định |
| Conflict handling | `upsert_fact` ghi đè; mệnh đề chứa `không còn`, `chỉ là`, `đùa`, `họp`, `lúc đầu` bị loại | Không giữ song song Huế/Đà Nẵng hay backend/MLOps | Danh sách cue viết tay, khó áp dụng cho cách diễn đạt khác |
| Entity extraction | `response_style` tách thành các slot `length/format/examples/emphasis`, merge theo slot; format mơ hồ không đè "3 bullet" | Giữ được "3 bullet" dù về sau người dùng chỉ nói "có cấu trúc" | Logic merge phức tạp hơn, cần test |
| Memory decay | `interests` xếp theo lần nhắc gần nhất, giữ tối đa 6 mục | Chặn file phình theo thời gian | Sở thích cũ nhưng vẫn đúng có thể bị đẩy ra |
| Grounding guardrail | `validate_fact_write()`: giá trị LLM ghi qua tool phải xuất hiện nguyên văn trong lời người dùng và vượt confidence threshold | Chặn đúng các giá trị sai Gemini đã ghi khi chạy live | Chặn cả fact đúng nếu LLM diễn đạt lại thay vì chép nguyên văn |

Giới hạn: phần trích fact offline dùng regex với danh sách thành phố và nghề cố định, chỉ phù hợp phạm vi dataset này.
