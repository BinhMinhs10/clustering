from doc_embedding.infer import EmbeddingModel
from doc_embedding.utils import prepare_features
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from transformers import AutoTokenizer, AutoModel
from pyvi.ViTokenizer import tokenize
import torch

tokenizer = AutoTokenizer.from_pretrained("data/checkpoint-7000")
model = AutoModel.from_pretrained("data/checkpoint-7000")

# emdedd = EmbeddingModel("data/sup-SimCSE-base")

sentence_0 = """Cơ quan điều tra xác định, Lê Anh Tuấn là chủ mưu và Trương Đình Vinh là đồng phạm trong vụ khống chế, ép em T. (16 tuổi, nhân viên quán karaoke tại xã Bình Thạnh) lên xe ô tô, chở đến khách sạn ở thị trấn Liên Hương để hiếp dâm.
Theo điều tra ban đầu, khoảng 17h chiều 31/10, Lê Anh Tuấn, Trương Đình Vinh, Nguyễn Văn Hùng (40 tuổi, quê Nghệ An), Nguyễn Văn Chiến (26 tuổi, quê Hà Tĩnh) và một người tên Sơn (chưa rõ lai lịch) đi trên 2 ô tô bán tải đến một quán karaoke tại xã Bình Thạnh ăn nhậu. 
Do T. là nhân viên phục vụ bưng bê trong quán, một người trong nhóm Tuấn đặt vấn đề "qua đêm" nhưng bị thiếu nữ này từ chối."""
sentence_1 = "Hành vi của bị cáo Phan Văn Vĩnh thể hiện sự bao che đến cùng"
sentence_2 = """Vũng Tàu nằm trong top 10 điểm đến được khách quốc tế tìm kiếm nhiều nhất Việt Nam
Đây là dữ liệu phân tích của công cụ Google Destination Insights. Theo Google Destination Insights, lượng tìm kiếm của du khách quốc tế về Việt Nam tăng dần từ đầu tháng 12/2021, đến cuối tháng 12/2021 thì tăng vọt. Lượt tìm kiếm vào thời điểm ngày 1/1 tăng 222% so với tháng trước và tăng 248% so với cùng kỳ 2021.
Lượng tìm kiếm nhiều nhất về du lịch Việt Nam đến từ các quốc gia như Mỹ, Úc, Nga, Pháp, Singapore, Ấn Độ, Nhật Bản, Đức, Anh, Canada… 10 điểm đến của Việt Nam được tìm kiếm nhiều nhất gồm có: TP.Hồ Chí Minh, Hà Nội, Nha Trang, Phú Quốc, Phan Thiết, Đà Nẵng, Hội An, Đà Lạt, Quy Nhơn, Vũng Tàu.
Khánh Hằng (TH)"""
sentence_3 = """
        Khách quốc tế tìm kiếm về du lịch Việt Nam nhiều nhất thế giới năm 2022
—
Theo báo cáo từ công cụ phân tích dữ liệu du lịch Google Destination Insights, lượng tìm kiếm của du khách quốc tế đối với hàng không và lưu trú du lịch Việt Nam tăng trưởng cao nhất trên toàn thế giới.
Nam miền Bắc
Tính từ đầu tháng 12/2021 đến nay, lượng tìm kiếm của du khách quốc tế đối với hàng không và cơ sở lưu trú du lịch của Việt Nam tăng trên 75%, mức tăng cao nhất trên toàn cầu. Xếp sau Việt Nam là các quốc gia như Chile, Bulgaria, New Zealand và Sri Lanka với tỷ lệ tăng từ 25% đến 75%.
Động lực chính mang lại tốc độ tăng trưởng đặc biệt ấn tượng của Việt Nam chính là nhu cầu tìm kiếm về hàng không.
Ngoài ra, Google Destination Insights cũng chỉ ra du khách đến từ các quốc gia như Mỹ, Nhật Bản, Ấn Độ, Đức, Anh, Brazil... có lượng tìm kiếm thông tin du lịch về Việt Nam nhiều nhất.
Mười điểm đến của Việt Nam được tìm kiếm nhiều nhất bao gồm: Sài Gòn, Đà Lạt, Phú Quốc, Vũng Tàu, Nha Trang, Hà Nội, Phan Thiết, Đà Nẵng, Hội An và Quy Nhơn.
Dù đóng cửa vì đại dịch Covid-19 gần 2 năm, hình ảnh Việt Nam vẫn lưu dấu ấn trên truyền thông quốc tế.
- Mới đây nhất, tạp chí du lịch nổi tiếng của Anh Wanderlust vừa đưa Việt Nam vào danh sách 20 điểm đến đáng đi nhất trong tháng 3. Tạp chí này đánh giá, tiết trời mùa xuân của tháng 3 là thời điểm lý tưởng nhất để khám phá những vùng đồi núi ở Tây Bắc Việt Nam. Nhiệt độ ổn định trong khi độ ẩm thấp, ít mưa sẽ là điều kiện thuận lợi để du khách thăm thú những bản làng xung quanh Sa Pa hoặc Mai Châu.
Tạp chí du lịch danh tiếng của Anh vinh danh Việt Nam là điểm đến lý tưởng trong tháng 3 này
Bên cạnh đó, Time Out, tạp chí toàn cầu chuyên về ẩm thực, du lịch, văn hóa, vừa qua cũng đánh giá, Hội An (Quảng Nam) là một trong những điểm đến lãng mạn nhất trên thế giới.
Có thể thấy, bất chấp thời gian dài tạm đóng cửa vì đại dịch, Việt Nam vẫn là điểm đến hấp dẫn với du khách quốc tế. Sự tăng trưởng về nhu cầu tìm kiếm quốc tế đối với hàng không và du lịch Việt Nam càng cho thấy tiềm năng của du lịch Việt Nam, là tín hiệu rất tích cực trước thềm ngành du lịch Việt Nam chuẩn bị mở cửa hoàn toàn từ ngày 15/3.
(Theo Google Insights)
        """

batch_sentences = [
    sentence_2, sentence_3
]

start = time.time()
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(device)
model = model.to(device)


def get_embeddings(sentences, model, tokenizer, device):
    sentences_tokenizer = [tokenize(sentence) for sentence in sentences]
    batch = prepare_features(sentences_tokenizer, tokenizer=tokenizer, max_len=256)

    # Move to the correct device
    for k in batch:
        batch[k] = torch.tensor(batch[k]).to(device)

    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.last_hidden_state.cpu()
        embeddings = pooler_output[batch['input_ids'] == tokenizer.mask_token_id]
    return embeddings


pooler_output = get_embeddings(batch_sentences, model, tokenizer, device)
print(pooler_output)

A = np.array(pooler_output[0])
B = np.array(pooler_output[1])
result = cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
print(result)
#
# topic_embeds = np.array(pooler_output)
# similarity_scores = topic_embeds.dot(A) / (np.linalg.norm(topic_embeds, axis=1) * np.linalg.norm(A))
# print(similarity_scores)

print("take time {}".format(time.time() - start))

