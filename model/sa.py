import torch
import torch.nn as nn
import torch.nn.functional as F

def full_attention_conv(q, k, v, output_attn=False):
    # Tính hằng số chuẩn hóa dựa trên số lượng Node (N)
    # n = q.shape[0] chính là số lượng hàng (tương ứng với số lượng Node) trong ma trận
    sqrt_n = torch.sqrt(torch.tensor(q.shape[0], dtype=torch.float32))

    # TÍNH MA TRẬN TƯƠNG QUAN ĐẶC TRƯNG
    # Đây là phép nhân K^T * V 
    # Ký hiệu einsum "lmh,ldh->mdh":
    # - l: Node index (sẽ bị triệt tiêu/cộng dồn sau dấu ->)
    # - m, d: Feature dimensions (giữ lại để tạo ma trận D x D)
    # - h: Head index (xử lý song song trên từng đầu attention)
    # Kết quả 'a' là ma trận [D, D, H] mô tả đặc trưng nào liên quan đến đặc trưng nào
    a = torch.einsum("lmh,ldh->mdh", k/sqrt_n, v)

    # Chuẩn hóa trọng số attention bằng Softmax trên chiều đặc trưng đầu tiên
    attention = torch.softmax(a, dim=0)

    # CẬP NHẬT THÔNG TIN CHO QUERY (Q)
    # Ký hiệu einsum "lmh,mdh->ldh":
    # - Lấy Query (mỗi node có D đặc trưng) nhân với bản đồ tương quan đặc trưng
    # - Kết quả 'output' trả về kích thước [N, D, H] ban đầu nhưng đã mang thông tin toàn cục
    output = torch.einsum("lmh,mdh->ldh", q, attention)

    if output_attn:
        return output, attention

    return output

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        # Khởi tạo ma trận trọng số cho Query và Key
        # Kích thước đầu ra được nhân với num_heads để xử lý đa đầu (Multi-head)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        
        # Ma trận trọng số cho Value (có thể tùy chọn dùng hoặc không)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels # Số chiều đặc trưng mỗi đầu
        self.num_heads = num_heads       # Số lượng đầu chú ý
        self.use_weight = use_weight

    def reset_parameters(self):
        # Khởi tạo lại trọng số ban đầu cho các lớp tuyến tính
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # BIẾN ĐỔI ĐẶC TRƯNG & CHIA HEADS
        # Chuyển đổi vector đặc trưng và chia nhỏ thành các khối (Heads)
        # .reshape(-1, D, H) -> [Số node, Số chiều mỗi head, Số head]
        query = self.Wq(query_input).reshape(-1, self.out_channels, self.num_heads)
        key = self.Wk(source_input).reshape(-1, self.out_channels, self.num_heads)
        
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.out_channels, self.num_heads)
        else:
            # Nếu không dùng trọng số cho Value, giữ nguyên đặc trưng nguồn
            value = source_input.reshape(-1, self.out_channels, 1)

        # TÍNH TOÁN LINEAR ATTENTION
        # Gọi hàm full_attention_conv để thực hiện Q(K^T * V)
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)
        else:
            attention_output = full_attention_conv(query, key, value)

        # HỢP NHẤT THÔNG TIN TỪ CÁC HEADS
        # Tính trung bình kết quả từ tất cả các đầu attention để tạo ra kết quả cuối cùng
        final_output = attention_output.mean(dim=-1)

        return (final_output, attn) if output_attn else final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation, num_layers=1, num_heads=2,
                 alpha=0.1, dropout=0.3, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()
        self.convs = nn.ModuleList() # Danh sách các tầng TransConvLayer
        self.fcs = nn.ModuleList() # Tầng Input Projection (MLP đầu vào)
        self.bns = nn.ModuleList() # Danh sách các lớp LayerNorm
        
        # Khởi tạo tầng chiếu đầu vào (Ví dụ: đưa 1433 chiều trong tập Cora về 256 chiều ẩn)
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        
        # Khởi tạo 'num_layers' tầng Attention liên tiếp
        for i in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
            
        self.dropout = dropout
        self.use_bn = use_bn
        self.residual = use_residual # Có sử dụng kết nối Residual hay không
        self.alpha = alpha # Hệ số trộn cho message: x = alpha * mới + (1-alpha) * cũ
        self.use_act = use_act
        self.activation = activation
        self.reset_parameters()

    def forward(self, x):
        layer_ = [] # Lưu trữ kết quả của từng tầng để phục vụ kết nối Residual
        
        # Chuyển đổi đặc trưng thô sang không gian ẩn (Hidden Dimension)
        x = self.fcs[0](x)
        if self.use_bn: x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x) # Lưu kết quả đầu tiên

        # Tiến hành lặp qua các tầng Attention ---
        for i, conv in enumerate(self.convs):
            # Tính toán mối quan hệ đặc trưng toàn cục
            x = conv(x, x)
            
            # Sử dụng Residual Connection giúp tránh mất thông tin khi mô hình có nhiều lớp
            if self.residual:
                # Phép trộn 'Combination' giữa đặc trưng vừa học và đặc trưng tầng trước
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            
            # Chuẩn hóa và áp dụng hàm kích hoạt
            if self.use_bn: x = self.bns[i + 1](x)
            if self.use_act: x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x) # Lưu kết quả tầng này cho tầng sau
            
        return x
    
    # Hàm hỗ trợ trích xuất ma trận Attention để phục vụ việc trực quan hóa
    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn: x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn: x = self.bns[i + 1](x)
            layer_.append(x)
        
        return torch.stack(attentions, dim=0)

