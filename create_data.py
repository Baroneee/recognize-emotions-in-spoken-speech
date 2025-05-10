import pandas as pd
import random

# Các mẫu câu đa dạng hơn cho từng nhãn
positive_samples = [
    "Mình cảm thấy vô cùng hạnh phúc khi được điểm cao trong bài kiểm tra.",
    "Bạn bè luôn động viên khiến mình tự tin hơn mỗi ngày.",
    "Thầy cô khen ngợi làm mình có động lực học tập hơn.",
    "Nhóm mình hợp tác rất ăn ý, kết quả vượt ngoài mong đợi.",
    "Mình vừa nhận được học bổng, cảm giác như mơ vậy!",
    "Bài thuyết trình hôm nay thành công rực rỡ, mình rất tự hào.",
    "Mình hoàn thành dự án trước thời hạn, cảm giác nhẹ nhõm và vui sướng.",
    "Mỗi ngày đến lớp đều là một trải nghiệm tuyệt vời.",
    "Bạn thân tặng quà sinh nhật bất ngờ, mình xúc động lắm.",
    "Mình được chọn làm lớp trưởng, cảm giác rất vinh dự.",
    "Cả lớp cùng nhau tổ chức tiệc, không khí thật vui vẻ.",
    "Mình nhận được lời khen từ thầy giáo chủ nhiệm.",
    "Bố mẹ tự hào về thành tích học tập của mình.",
    "Mình giúp bạn giải bài tập khó, cả hai đều vui.",
    "Tham gia câu lạc bộ mới, mình quen được nhiều bạn tốt.",
    "Mình đạt giải nhất cuộc thi hùng biện của trường.",
    "Bạn cùng lớp hỗ trợ nhau học tập rất nhiệt tình.",
    "Mình được mời tham gia đội tuyển học sinh giỏi.",
    "Cảm giác chiến thắng khi vượt qua thử thách lớn.",
    "Mình vừa học xong một bài mới và hiểu rất rõ."
]

negative_samples = [
    "Mình cảm thấy thất vọng vì bị điểm kém trong bài kiểm tra.",
    "Bạn bè không quan tâm khiến mình thấy cô đơn.",
    "Thầy cô phê bình làm mình buồn cả ngày.",
    "Nhóm mình bất đồng ý kiến, kết quả không như mong đợi.",
    "Mình bị mất đồ dùng học tập, cảm giác rất tệ.",
    "Bài thuyết trình hôm nay gặp sự cố, mình lo lắng mãi.",
    "Mình nộp bài trễ nên bị trừ điểm, rất buồn.",
    "Mỗi ngày đến lớp đều cảm thấy áp lực.",
    "Bạn thân hiểu lầm khiến mình buồn bã.",
    "Mình bị loại khỏi đội tuyển, cảm giác thất vọng.",
    "Cả lớp bị phạt vì một lỗi nhỏ, mình thấy bất công.",
    "Mình bị điểm danh nhầm, không biết giải thích sao.",
    "Bố mẹ không hài lòng về kết quả học tập của mình.",
    "Mình không làm được bài kiểm tra, cảm giác bất lực.",
    "Tham gia câu lạc bộ nhưng không ai nói chuyện với mình.",
    "Mình bị chê bai trước lớp, rất xấu hổ.",
    "Bạn cùng lớp không hợp tác khi làm nhóm.",
    "Mình bị đau ốm nên không thể đi học.",
    "Cảm giác thất bại khi không đạt mục tiêu đề ra.",
    "Mình vừa bị điểm kém môn Toán, rất buồn."
]

neutral_samples = [
    "Hôm nay mình đi học như thường lệ, không có gì đặc biệt.",
    "Lớp học diễn ra bình thường, mọi người đều chăm chú nghe giảng.",
    "Mình vừa hoàn thành bài tập về nhà.",
    "Thời tiết hôm nay mát mẻ, mình đi học đúng giờ.",
    "Giáo viên giao bài tập mới, mình sẽ làm vào cuối tuần.",
    "Mình tham gia buổi học nhóm, mọi thứ diễn ra bình thường.",
    "Hôm nay mình đọc tài liệu môn học.",
    "Lịch học hôm nay khá dày, mình cần sắp xếp thời gian.",
    "Mình vừa làm xong bài tập nhỏ.",
    "Ngày mai mình có bài kiểm tra, cần ôn lại một chút.",
    "Mình ăn trưa ở căn tin cùng bạn bè.",
    "Giờ ra chơi, mình ngồi đọc sách.",
    "Mình đến lớp đúng giờ và nghe giảng đầy đủ.",
    "Hôm nay không có sự kiện gì nổi bật ở trường.",
    "Mình chuẩn bị sách vở cho buổi học tiếp theo.",
    "Mình ghi chú lại bài giảng để dễ học hơn.",
    "Mình làm bài tập nhóm cùng các bạn.",
    "Mình kiểm tra lại lịch học tuần này.",
    "Mình vừa nhận được thông báo từ giáo viên.",
    "Mình tham gia tiết học thể dục như mọi khi."
]

# Hàm tạo câu ngẫu nhiên phức tạp hơn
def augment_sentence(sentence):
    adverbs = ["thật sự", "rất", "cực kỳ", "khá", "hơi", "quá", "vô cùng", "đặc biệt"]
    situations = [
        "trong lớp học", "ở trường", "khi làm bài tập", "trong giờ ra chơi", "khi thảo luận nhóm",
        "vào buổi sáng", "vào buổi chiều", "trong kỳ thi", "khi nói chuyện với bạn bè"
    ]
    endings = [
        ".", "!", " :)", " và mình nghĩ sẽ tốt hơn lần sau.", " - mình tự nhủ.", " (thật bất ngờ)."
    ]
    # Thêm trạng từ, tình huống, kết thúc ngẫu nhiên
    s = sentence
    if random.random() < 0.5:
        s = random.choice(adverbs) + " " + s
    if random.random() < 0.5:
        s = s + " " + random.choice(situations)
    if random.random() < 0.7:
        s = s + random.choice(endings)
    return s

data = []
# 333 positive, 333 negative, 334 neutral (tổng 1000)
for i in range(333):
    text = augment_sentence(random.choice(positive_samples))
    data.append((text, "Positive"))

for i in range(333):
    text = augment_sentence(random.choice(negative_samples))
    data.append((text, "Negative"))

for i in range(334):
    text = augment_sentence(random.choice(neutral_samples))
    data.append((text, "Neutral"))

random.shuffle(data)

print(f"Tổng số mẫu: {len(data)}")
print(f"Positive: {sum(1 for _, label in data if label == 'Positive')}")
print(f"Negative: {sum(1 for _, label in data if label == 'Negative')}")
print(f"Neutral: {sum(1 for _, label in data if label == 'Neutral')}")

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("sentiment_data.csv", index=False)
print("Đã tạo file sentiment_data.csv với 1000 mẫu")