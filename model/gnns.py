from typing import Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Lớp thực hiện phép tích chập đồ thị cục bộ 
# Trong DUALFormer, đây chính là phần xử lý trên chiều node
class Graph_Conv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        improved: bool = False,
        cached: bool = False, # Nếu True, ma trận kề chuẩn hóa sẽ được lưu lại để dùng cho lần sau
        add_self_loops: Optional[bool] = None, # Tự thêm kết nối với chính mình (self-loop)
        normalize: bool = True, # Chuẩn hóa ma trận kề (rất quan trọng trong GCN/SGC)
        bias: bool = True,
        **kwargs,
    ):
        # Thiết lập hàm Aggregation (Tổng hợp) mặc định là phép cộng ('add')
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Các biến dùng để lưu trữ ma trận kề sau khi đã chuẩn hóa (giúp tăng tốc độ)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        # Làm mới các tham số và xóa bộ nhớ đệm ma trận kề
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        # CHUẨN HÓA MA TRẬN KỀ 
        # Đây là bước tiền xử lý để làm mịn dữ liệu đồ thị, giúp tránh hiện tượng gradient bị bùng nổ.
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    # Tính toán ma trận chuẩn hóa D^-1/2 * A * D^-1/2
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim))
                    # Nếu bật cached, ta lưu lại kết quả này để các vòng lặp sau không phải tính lại
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                # Xử lý tương tự nếu dữ liệu đầu vào là dạng ma trận thưa (SparseTensor)
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim))
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # LAN TRUYỀN THÔNG TIN 
        # Hàm propagate sẽ kích hoạt chuỗi: message -> aggregate -> update
        # Trong DUALFormer, bước này thu thập thông tin từ hàng xóm (Local Context)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    # Định nghĩa Message gửi đi từ node láng giềng j đến node đích i
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # Đặc trưng của hàng xóm (x_j) được nhân với trọng số cạnh đã chuẩn hóa (edge_weight)
        # Nếu không có trọng số, tin nhắn giữ nguyên. Nếu có, nó sẽ được cân chỉnh.
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # Hàm tối ưu hóa kết hợp giữa gửi tin và tổng hợp (thường dùng trong ma trận thưa)
    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)