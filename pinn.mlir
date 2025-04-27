module @reactant_Physics... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<32x3xf32>, %arg1: tensor<3x32xf32>, %arg2: tensor<32xf32>, %arg3: tensor<32x32xf32>, %arg4: tensor<32xf32>, %arg5: tensor<32x32xf32>, %arg6: tensor<32xf32>, %arg7: tensor<32x3xf32>, %arg8: tensor<3xf32>) -> tensor<32x3xf32> {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<3x32xf32>, tensor<32x3xf32>) -> tensor<32x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0] : (tensor<32xf32>) -> tensor<32x32xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x32xf32>
    %3 = stablehlo.tanh %2 : tensor<32x32xf32>
    %4 = stablehlo.dot_general %arg3, %3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [0] : (tensor<32xf32>) -> tensor<32x32xf32>
    %6 = stablehlo.add %4, %5 : tensor<32x32xf32>
    %7 = stablehlo.tanh %6 : tensor<32x32xf32>
    %8 = stablehlo.dot_general %arg5, %7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = stablehlo.broadcast_in_dim %arg6, dims = [0] : (tensor<32xf32>) -> tensor<32x32xf32>
    %10 = stablehlo.add %8, %9 : tensor<32x32xf32>
    %11 = stablehlo.tanh %10 : tensor<32x32xf32>
    %12 = stablehlo.dot_general %11, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x3xf32>) -> tensor<32x3xf32>
    %13 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<3xf32>) -> tensor<32x3xf32>
    %14 = stablehlo.add %12, %13 : tensor<32x3xf32>
    return %14 : tensor<32x3xf32>
  }
}