import torch
import torch.nn.functional as F
from torch.nn import Linear
from model.gnns import Graph_Conv
from model.sa import TransConv
from torch_geometric.nn import GCNConv, APPNP

# Lớp chính của mô hình DUALFormer - Kiến trúc Graph Transformer lai
class DUALFormer_Model(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 output_dim,
                 activation,
                 num_gnns,
                 num_trans,
                 num_heads,
                 dropout_trans,
                 dropout,
                 alpha,
                 use_bn,
                 lammda=0.1,
                 GraphConv='sgc'):
        super(DUALFormer_Model, self).__init__()
        
        # Khởi tạo hàm kích hoạt (ReLU, ELU, PReLU...) tùy chọn từ config
        self.activation = activation()
        self.num_gnns = num_gnns
        
        # Khởi tạo Module GLOBAL ATTENTION (hoạt động trên chiều đặc trưng DxD)
        # Đây là module khiến DUALFormer trở nên nổi trội. Module này xử lý các mối quan hệ toàn cục.
        # Khác với Transformer thường, ở đây dùng 'Linear Attention' nên độ phức tạp thấp.
        self.layers_trans = TransConv(input_dim, hidden_dim, self.activation,
                                      num_layers=num_trans, # Số lớp SA (đối với dataset Cora là 3 lớp)
                                      num_heads=num_heads, # Số đầu chú ý
                                      alpha=alpha, # Tham số 'Combination' để trộn tin cũ và mới
                                      dropout=dropout_trans,
                                      use_bn=use_bn, # Dùng LayerNorm để ổn định training
                                      use_residual=True, # Kết nối Residual giúp chống 'Over-smoothing' ở các lớp sâu
                                      use_weight=True, use_act=True)

        # Module LOCAL GNN (hoạt động dựa trên cấu trúc liên kết trong đồ thị)
        # Sau khi đã có biểu diễn toàn cục từ bước trên, ta sẽ dùng các lớp GNN để trích xuất thông tin cục bộ.
        # Bước này tương ứng với phép 'Aggregation' trong lý thuyết GNN.
        if GraphConv == 'sgc':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                # SGC được tác giả lựa chọn làm mặc định vì nhẹ và hiệu quả trên tập Cora
                self.convs.append(Graph_Conv())
        elif GraphConv == 'gcn':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif GraphConv == 'appnp':
            self.convs = APPNP(num_gnns, lammda)

        self.GraphConv = GraphConv
        
        # Chuyển đặc trưng ẩn về số lớp cần phân loại (ví dụ từ 256 chiều ẩn thành 7 lớp của Cora)
        self.linear_project = Linear(hidden_dim, output_dim)
        self.dropout = dropout

        # Phân tách tham số để optimizer có thể áp dụng Weight Decay khác nhau cho từng phần
        self.params1 = list(self.layers_trans.parameters()) # Nhóm tham số của module Global Attention và module MLP đầu vào
        self.params2 = list(self.linear_project.parameters()) # Nhóm tham số của module GNN và lớp cuối

        self.traning = True
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo lại trọng số ban đầu để đảm bảo tính ngẫu nhiên mỗi lần chạy
        self.layers_trans.reset_parameters()
        self.linear_project.reset_parameters()

    def forward(self, x, edge_index):

        # BƯỚC 1: Xử lý toàn cục (Global Attention)
        # Input đi qua Input Projection Layer (ẩn trong TransConv) rồi tính toán Attention.
        # Ở đây code thực hiện Q(K^T V) giúp độ phức tạp chỉ là O(N).
        z = self.layers_trans(x)
        
        # BƯỚC 2: Xử lý cấu trúc cục bộ (Local GNN)
        # Lấy kết quả từ bước thực hiện truyền tin qua các cạnh của đồ thị.
        if self.GraphConv in ['sgc', 'gcn']:#sgc, gcn
            for i, conv in enumerate(self.convs):
                z = conv(z, edge_index)
        else:
            z = self.convs(z, edge_index) #appnp
            
        # Áp dụng Dropout để chống 'Overfitting' (Đặc biệt quan trọng với tập dữ liệu nhỏ)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.linear_project(z)

        # Chiếu về không gian nhãn và dùng Log-Softmax để lấy xác suất phân loại
        return F.log_softmax(z, dim=1)
