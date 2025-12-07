
import torch
from vllm.model_executor.layers.linear import DecomposedLinear, ReplicatedLinear

def test_decomposed_linear():
    input_size = 16
    mid_size = 8
    output_size = 4
    batch_size = 2

    # Instantiate with ReplicatedLinear to avoid distributed setup requirements
    layer = DecomposedLinear(
        input_size=input_size,
        output_size=output_size,
        mid_size=mid_size,
        pre_layer_cls=ReplicatedLinear,
        post_layer_cls=ReplicatedLinear,
        disable_tp=True
    )

    # Initialize weights for testing
    # a: [mid, input] (PyTorch convention) -> [8, 16]
    # b: [output, mid] -> [4, 8]
    # ReplicatedLinear creates weight of shape [output, input]
    
    # Check shapes
    print(f"Layer a weight shape: {layer.a.weight.shape}")
    print(f"Layer b weight shape: {layer.b.weight.shape}")
    
    assert layer.a.weight.shape == (mid_size, input_size)
    assert layer.b.weight.shape == (output_size, mid_size)

    x = torch.randn(batch_size, input_size)
    output = layer(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_size)
    
    # Verify logic: y = x @ A.T @ B.T + bias
    # layer.a.weight is A (mid, input)
    # layer.b.weight is B (output, mid)
    # layer.b.bias is bias (output)
    
    # a(x) = x @ A.T
    res_a = torch.matmul(x, layer.a.weight.t())
    # b(res_a) = res_a @ B.T + bias
    expected = torch.matmul(res_a, layer.b.weight.t()) + layer.b.bias
    
    if torch.allclose(output, expected, atol=1e-5):
        print("Verification passed!")
    else:
        print("Verification failed!")
        print("Output:", output)
        print("Expected:", expected)

if __name__ == "__main__":
    test_decomposed_linear()
