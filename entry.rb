require 'matrix'
require 'chunky_png'

XOR             = ->(a,b) {a^b}
SIGMOID         = ->(a) {1/(1+Math.exp(-a))}
PD_SIGMOID      = ->(a) {a*(1-a)}
ERROR           = ->(exp,out) {0.5*((exp-out)**2)}
PD_ERROR        = ->(exp,out) {-(exp-out)}
DELTA_RULE      = ->(a,b,c) {a*b*c}
DELTA_WEIGHT    = ->(a,b) {a-(0.5*b)} # 0.5 = learning rate

# input layer has 3 nodes for XOR; 1 Bias + 2 Inputs
# hidden layer will have 4 nodes; 1 Bias + 3 Inputs
# randomise all of the weights. 9 = 3 inputs * (4-1) hidden. The Bias does not count
$ih_weights = Array.new(3) {Array.new(3,rand)}
$ho_weights = Array.new(1) {Array.new(4,rand)} # 4 hidden * 1 output - Not a matrix since only 1 output
$i_nodes = [999,999,1] # 1 here is the Bias
$h_nodes = [999,999,999,1] # 1 here is the Bias
$o_nodes = [0]

# This is not needed. It was implemented as a validator to see that images are read correctly
def convert_to_images(arr,rows,cols)
    canvas = ChunkyPNG::Canvas.new(28, 28, ChunkyPNG::Color::TRANSPARENT)
    canvas.grayscale!
    
    for i in 0..arr.length-1 do
        for j in 0..rows-1 do
            for k in 0..cols-1 do
                canvas[j,k] = arr[i][((cols+1)*j)+k]
            end
        end
        canvas.save("F:\\Projects\\YetAnotherAnn\\filename#{i}.png", :interlace => true)
    end
end

def feed_forward
    # compute the hidden node values
    for i in 0..$h_nodes.length-2 do
        sum = 0
        for j in 0..$i_nodes.length-1 do
            sum += ($ih_weights[i][j] * $i_nodes[j])
        end
        $h_nodes[i] = SIGMOID.call sum
    end

    # contribute these to the final output
    for i in 0..$o_nodes.length-1 do
        sum = 0
        for j in 0..$h_nodes.length-1 do
            sum += ($ho_weights[i][j] * $h_nodes[j])
        end
        $o_nodes[i] = SIGMOID.call sum
    end
end

def train(input1, input2, expected)
    $i_nodes[0] = input1
    $i_nodes[1] = input2

    feed_forward

    ########################################################
    # BACK PROPAGATION
    new_weights_hidden = []
    for i in 0..$o_nodes.length-1 do
        pd_error_output = PD_ERROR.call expected[i],$o_nodes[i]
        pd_sigmoid_output = PD_SIGMOID.call $o_nodes[i]

        # new weights for hidden layer
        new_weights_hidden[i] = []
        for j in 0..$ho_weights[i].length - 1 do
            delta_rule_h = DELTA_RULE.call pd_error_output,pd_sigmoid_output,$h_nodes[j]
            new_weights_hidden[i][j] = DELTA_WEIGHT.call $ho_weights[i][j],delta_rule_h
        end
    end

    # do not include the Bias!
    new_weights_input = Array.new(3) {Array.new(3,0)}
    for i in 0..$h_nodes.length - 2 do # 2 because of the Bias node. this has no weights attached to it
        error_output_h = 0
        for j in 0..$ho_weights.length-1 do
            pd_error_output = PD_ERROR.call expected[j],$o_nodes[j]
            pd_sigmoid_output = PD_SIGMOID.call $o_nodes[j]

            error_over_net = pd_error_output * pd_sigmoid_output
            error_output_h += (error_over_net * $ho_weights[j][i]) # 0.036350306
        end
        pd_sigmoid_h_node = PD_SIGMOID.call $h_nodes[i] # 0.241300709

        for j in 0..$i_nodes.length-1 do
            delta_rule_i = DELTA_RULE.call error_output_h,pd_sigmoid_h_node,$i_nodes[j]
            new_weights_input[i][j] = DELTA_WEIGHT.call $ih_weights[i][j],delta_rule_i
        end
    end

    # update the weights
    $ho_weights = new_weights_hidden
    $ih_weights = new_weights_input
    # BACK PROPAGATION
    ########################################################
end

def read_mnist_file_images(path)
    file = File.open(path, "rb") # opens file in binary format

    # first 16 bytes are 32-bit integer in Big Endian format
    magic = ((file.read 4).unpack 'N' * 4).first    
    amount = ((file.read 4).unpack 'N' * 4).first
    rows = ((file.read 4).unpack 'N' * 4).first
    cols = ((file.read 4).unpack 'N' * 4).first

    arr = []
    for i in 0..3 do #amount-1 do
        tmp_arr = []
        for j in 0..rows-1 do
            for k in 0..cols-1 do
                tmp_arr[((cols+1)*j)+k] = ((file.read 1).unpack 'C' * 1).first
            end
        end
        arr[i] = tmp_arr
    end
    convert_to_images arr,28,28
    arr
end

def read_mnist_file_labels(path)
    file = File.open(path, "rb") # opens file in binary format
    magic = ((file.read 4).unpack 'N' * 4).first    
    amount = ((file.read 4).unpack 'N' * 4).first

    arr = []
    for i in 0..3 do #amount-1 do
        arr[i] = ((file.read 1).unpack 'C' * 1).first
    end
    arr
end

images = read_mnist_file_images "#{__dir__}\\data\\train-images.idx3-ubyte"
labels = read_mnist_file_labels "#{__dir__}\\data\\train-labels.idx1-ubyte"