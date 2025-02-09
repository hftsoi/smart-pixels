import sympy
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import Zeros, RandomNormal
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers, models, Model

np.random.seed(42)
tf.random.set_seed(42)

def math_operation(tf_or_sympy, operator, x, y=None):
    # unary operators
    if operator == 'identity':
        output = x
    elif operator == 'sin':
        output = tf.sin(x) if tf_or_sympy == 'tf' else sympy.sin(x)
    elif operator == 'cos':
        output = tf.cos(x) if tf_or_sympy == 'tf' else sympy.cos(x)
    elif operator == 'exp':
        output = tf.exp(x) if tf_or_sympy == 'tf' else sympy.exp(x)
    elif operator == 'gauss':
        output = tf.exp(-x**2) if tf_or_sympy == 'tf' else sympy.exp(-x**2)
    elif operator == 'sinh':
        output = tf.sinh(x) if tf_or_sympy == 'tf' else sympy.sinh(x)
    elif operator == 'cosh':
        output = tf.cosh(x) if tf_or_sympy == 'tf' else sympy.cosh(x)
    elif operator == 'tanh':
        output = tf.tanh(x) if tf_or_sympy == 'tf' else sympy.tanh(x)
    elif operator == 'square':
        output = x**2 if tf_or_sympy == 'tf' else x**2
    elif operator == 'cube':
        output = x**3 if tf_or_sympy == 'tf' else x**3
    elif operator == 'log':
        output = tf.math.log(0.001 + tf.abs(x)) if tf_or_sympy == 'tf' else sympy.log(0.001 + sympy.Abs(x))
        #output = tf.math.log(tf.abs(x)) if tf_or_sympy == 'tf' else sympy.log(sympy.Abs(x))
    # binary operators
    elif operator == '+':
        output = x + y
    elif operator == '*':
        output = x * y
    elif operator == 'pow':
        output = x ** y
    elif operator == '/':
        output = x / (0.001 + tf.abs(y)) if tf_or_sympy == 'tf' else x / (0.001 + sympy.Abs(y))
        #output = x / y
    
    return output

# step function for masking in the forward pass
# custom grad is an estimator of derivative of the step function to avoid vanishing gradient
@tf.custom_gradient
def step_func(x):
    func = tf.where(x > 0., 1., 0.)
    def grad(upstream):
        a = 5.
        return upstream * a * tf.exp(-a*x) / (1 + tf.exp(-a*x))**2
    return func, grad

# dynamic pruning of input features
# define one auxiliary weight and one untrainable threshold per input feature
class Input_sparsity(Layer):
    def __init__(self):
        super().__init__()
        
    def build(self, input_shape):
        # auxiliary weight is untrainable and fixed at 1
        self.aux_w = self.add_weight(name='weight',
                                     shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=False)
        # threshold is trainable, initialized at 0, and bounded in [0,1]
        self.aux_w_t = self.add_weight(name='threshold',
                                     shape=(input_shape[-1],),
                                     initializer='zeros',
                                     constraint=lambda x: tf.clip_by_value(x, 0., 0.5), # input prunning turned off!!!
                                     trainable=True)
        
    def call(self, inputs):
        input_masks = step_func(self.aux_w - self.aux_w_t)
        return tf.multiply(inputs, input_masks)

# definition of symbolic layer (usual weights and biases, plus unary/binary operators as activations)
# define trainable thresholds for weights/biases/unary/binary
class Symbolic_Layer(Layer):
    def __init__(self, operators, num_operators):
        super().__init__()
        self.operators = operators
        self.num_operators = num_operators
        self.units = self.num_operators[0] + 2*self.num_operators[1]
        
    def build(self, input_shape):
        # usual model weight w
        self.w = self.add_weight(name='weight',
                                 shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        # usual bias b
        self.b = self.add_weight(name='bias',
                                 shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        # auxiliary weight for unary operator, to be used for operator pruning
        # untrainable and fixed at 1
        self.aux_unary = self.add_weight(name='unary',
                                         shape=(self.num_operators[0],),
                                         initializer='ones',
                                         trainable=False)
        # auxiliary weight for binary operator, to be used for operator pruning
        # untrainable and fixed at 1
        if self.num_operators[1] > 0:
            self.aux_binary = self.add_weight(name='binary',
                                              shape=(self.num_operators[1],),
                                              initializer='ones',
                                              trainable=False)
        # threshold for model weight
        # trainable, initialized at 0, unbounded since model weight is unbounded
        self.aux_w_t = self.add_weight(name='weight_threshold',
                                       shape=(input_shape[-1], self.units),
                                       initializer='zeros',
                                       constraint=lambda x: tf.abs(x),
                                       trainable=True)
        # threshold for bias term
        # trainable, initialized at 0, unbounded since model weight is unbounded
        self.aux_b_t = self.add_weight(name='bias_threshold',
                                       shape=(self.units,),
                                       initializer='zeros',
                                       constraint=lambda x: tf.abs(x),
                                       trainable=True)
        # threshold for unary operator
        # trainable, initialized at 0, bounded in [0,1]
        self.aux_unary_t = self.add_weight(name='unary_threshold',
                                          shape=(self.num_operators[0],),
                                          initializer='zeros',
                                          constraint=lambda x: tf.clip_by_value(x, 0., 1.),
                                          trainable=True)
        # threshold for binary operator
        # trainable, initialized at 0, bounded in [0,1]
        if self.num_operators[1] > 0:
            self.aux_binary_t = self.add_weight(name='binary_threshold',
                                                shape=(self.num_operators[1],),
                                                initializer='zeros',
                                                constraint=lambda x: tf.clip_by_value(x, 0., 1.),
                                                trainable=True)

    def call(self, inputs):
        # linear transformation in the forward pass
        # weights and biases are replaced by the step_functioned version
        # so weight is masked whenever its threshold is higher
        w_masks = step_func(tf.abs(self.w) - self.aux_w_t)
        b_masks = step_func(tf.abs(self.b) - self.aux_b_t)
        linear_output = tf.matmul(inputs, tf.multiply(self.w, w_masks)) + tf.multiply(self.b, b_masks)
        
        # activation by unary/binary operator
        symbolic_output = []
        
        # loop over number of unary operators in a symbolic layer
        for i in range(self.num_operators[0]):
            # an unary operator is "pruned" (becomes identity map) whenever the threshold is higher than its auxiliary weight
            unary_mask = step_func(self.aux_unary - self.aux_unary_t)[i]
            idx = np.mod(i, len(self.operators[0]))
            unary_operation = (unary_mask * math_operation('tf',self.operators[0][idx],linear_output[:, i:i+1]) +
                               (1.0 - unary_mask) * math_operation('tf','identity',linear_output[:, i:i+1]))
            symbolic_output.append(unary_operation)
            
        # loop over number of binary operators in a symbolic layer
        for i in range(self.num_operators[0], self.num_operators[0] + 2*self.num_operators[1], 2):
            # a binary operator is "pruned" (becomes addition) whenever the threshold is higher than its auxiliary weight
            j = int((i - self.num_operators[0])/2)
            binary_mask = step_func(self.aux_binary - self.aux_binary_t)[j]
            idx = np.mod(j, len(self.operators[1]))
            binary_operation = (binary_mask * math_operation('tf',self.operators[1][idx],linear_output[:, i:i+1],linear_output[:, i+1:i+2]) +
                                (1.0 - binary_mask) * math_operation('tf','+',linear_output[:, i:i+1],linear_output[:, i+1:i+2]))
            symbolic_output.append(binary_operation)
        
        symbolic_output = tf.concat(symbolic_output, axis=1)
        return symbolic_output
    
def create_model(model_dim, operators, num_operators):
    input_dim, num_hidden_layers, output_dim = model_dim
    layers = []
    
    # input layer
    layers.append(Input(shape=(input_dim,)))
    layers.append(Input_sparsity()(layers[-1]))
    
    # hidden symbolic layers
    for i in range(num_hidden_layers):
        layers.append(Symbolic_Layer(operators=operators[i],
                                     num_operators=num_operators[i])(layers[-1]))
    
    # output layer
    layers.append(Symbolic_Layer(operators=[['identity'], [None]],
                                 num_operators=[output_dim, 0])(layers[-1]))
    
    model = keras.Model(inputs=layers[0], outputs=layers[-1], name='model')
    return model, model_dim, operators, num_operators


class neuralSR(keras.Model):
    def __init__(self, model, alpha_sparsity_input, alpha_sparsity_model, alpha_sparsity_unary, alpha_sparsity_binary):
        super().__init__()
        self.model, self.model_dim, self.operators, self.num_operators = model
        self.alpha_sparsity_input = alpha_sparsity_input
        self.alpha_sparsity_model = alpha_sparsity_model
        self.alpha_sparsity_unary = alpha_sparsity_unary
        self.alpha_sparsity_binary = alpha_sparsity_binary
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.regression_loss_tracker = keras.metrics.Mean(name="regression_loss")
        self.threshold_input_reg_loss_tracker = keras.metrics.Mean(name="threshold_input_reg_loss")
        self.threshold_input_mean_tracker = keras.metrics.Mean(name="threshold_input_mean")
        self.threshold_model_reg_loss_tracker = keras.metrics.Mean(name="threshold_model_reg_loss")
        self.threshold_model_mean_tracker = keras.metrics.Mean(name="threshold_model_mean")
        self.threshold_unary_reg_loss_tracker = keras.metrics.Mean(name="threshold_unary_reg_loss")
        self.threshold_unary_mean_tracker = keras.metrics.Mean(name="threshold_unary_mean")
        self.threshold_binary_reg_loss_tracker = keras.metrics.Mean(name="threshold_binary_reg_loss")
        self.threshold_binary_mean_tracker = keras.metrics.Mean(name="threshold_binary_mean")
        self.weight_input_mean_tracker = keras.metrics.Mean(name="weight_input_mean")
        self.weight_model_mean_tracker = keras.metrics.Mean(name="weight_model_mean")
        self.weight_unary_mean_tracker = keras.metrics.Mean(name="weight_unary_mean")
        self.weight_binary_mean_tracker = keras.metrics.Mean(name="weight_binary_mean")
        self.sparsity_input_tracker = keras.metrics.Mean(name="sparsity_input")
        self.sparsity_model_tracker = keras.metrics.Mean(name="sparsity_model")
        self.sparsity_unary_tracker = keras.metrics.Mean(name="sparsity_unary")
        self.sparsity_binary_tracker = keras.metrics.Mean(name="sparsity_binary")
        self.accuracy_tracker = keras.metrics.Accuracy(name="accuracy")
    
    def get_hyperparameters(self):
        return self.model_dim, self.operators, self.num_operators
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.regression_loss_tracker,
            self.threshold_input_reg_loss_tracker,
            self.threshold_input_mean_tracker,
            self.threshold_model_reg_loss_tracker,
            self.threshold_model_mean_tracker,
            self.threshold_unary_reg_loss_tracker,
            self.threshold_unary_mean_tracker,
            self.threshold_binary_reg_loss_tracker,
            self.threshold_binary_mean_tracker,
            self.weight_input_mean_tracker,
            self.weight_model_mean_tracker,
            self.weight_unary_mean_tracker,
            self.weight_binary_mean_tracker,
            self.sparsity_input_tracker,
            self.sparsity_model_tracker,
            self.sparsity_unary_tracker,
            self.sparsity_binary_tracker,
            self.accuracy_tracker
        ]
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            
            # base training loss (MSE)
            regression_loss = tf.reduce_mean(tf.reduce_sum(tf.cast((y-y_pred)**2,dtype=tf.float64), axis=1))
            
            # the followings calculate sparsity levels at train steps
            # since total loss = MSE + sparsity regularization terms
            # where sparsity regularization terms depend on sparsity levels
            
            # call auxiliary weights and thresholds for inputs
            w_input = self.non_trainable_weights[0]
            w_t_input = self.trainable_weights[0]
            num_input_masks = tf.reduce_sum(tf.cast(tf.where(w_input - w_t_input > 0., 0., 1.), dtype=tf.float64))
            num_input_weights = tf.size(w_input, out_type=tf.float64)
            sparsity_input = num_input_masks/num_input_weights
            weight_input_mean = tf.reduce_sum(tf.cast(tf.abs(w_input),dtype=tf.float64))/num_input_weights
            
            # calculate input sparsity
            t_input_sum = tf.reduce_sum(tf.cast(w_t_input,dtype=tf.float64))
            t_input_dim = num_input_weights
            threshold_input_mean = t_input_sum/t_input_dim
            
            # call model weights and thresholds for all hidden symbolic layers
            # then calculate model weight (weight+bias) sparsity
            num_model_masks = 0.
            num_model_weights = 0.
            weight_model_mean = 0.
            t_model_sum = 0.
            t_model_dim = 0.
            sum_exp_t = 0.
            for i in range(self.model_dim[1]+1):
                w_model = self.trainable_weights[1+6*i]
                b_model = self.trainable_weights[1+6*i+1]
                w_t_model = self.trainable_weights[1+6*i+2]
                b_t_model = self.trainable_weights[1+6*i+3]
                sum_exp_t += tf.reduce_sum(tf.exp(-tf.cast(w_t_model,dtype=tf.float64)))
                sum_exp_t += tf.reduce_sum(tf.exp(-tf.cast(b_t_model,dtype=tf.float64)))
                num_model_masks += tf.reduce_sum(tf.cast(tf.where(-w_t_model + tf.abs(w_model) > 0., 0., 1.),dtype=tf.float64))
                num_model_masks += tf.reduce_sum(tf.cast(tf.where(-b_t_model + tf.abs(b_model) > 0., 0., 1.),dtype=tf.float64))
                num_model_weights += tf.size(w_model, out_type=tf.float64)
                num_model_weights += tf.size(b_model, out_type=tf.float64)
                weight_model_mean += tf.reduce_sum(tf.cast(tf.abs(w_model),dtype=tf.float64)) + tf.reduce_sum(tf.cast(tf.abs(b_model),dtype=tf.float64))
                t_model_dim += tf.size(w_t_model, out_type=tf.float64) + tf.size(b_t_model, out_type=tf.float64)
                t_model_sum += tf.reduce_sum(tf.cast(w_t_model,dtype=tf.float64)) + tf.reduce_sum(tf.cast(b_t_model,dtype=tf.float64))
            sparsity_model = num_model_masks/num_model_weights
            weight_model_mean = weight_model_mean/num_model_weights

            threshold_model_mean = t_model_sum/t_model_dim
            
            # call auxiliary weights and thresholds for unary and binary operators
            num_unary_masks = 0.
            num_unary_weights = 0.
            weight_unary_mean = 0.
            num_binary_masks = 0.
            num_binary_weights = 0.
            weight_binary_mean = 0.
            t_u_sum = 0.
            t_u_dim = 0.
            t_b_sum = 0.
            t_b_dim = 0.
            for i in range(self.model_dim[1]):
                u = self.non_trainable_weights[2*i+1]
                u_t = self.trainable_weights[1+6*i+4]
                num_unary_masks += tf.reduce_sum(tf.cast(tf.where(u - u_t > 0., 0., 1.),dtype=tf.float64))
                num_unary_weights += tf.size(u, out_type=tf.float64)
                weight_unary_mean += tf.reduce_sum(tf.cast(u,dtype=tf.float64))
                t_u_dim += tf.size(u_t, out_type=tf.float64)
                t_u_sum += tf.reduce_sum(tf.cast(u_t,dtype=tf.float64))
                
                b = self.non_trainable_weights[2*i+2]
                b_t = self.trainable_weights[1+6*i+5]
                num_binary_masks += tf.reduce_sum(tf.cast(tf.where(b - b_t > 0., 0., 1.),dtype=tf.float64))
                num_binary_weights += tf.size(b, out_type=tf.float64)
                weight_binary_mean += tf.reduce_sum(tf.cast(b,dtype=tf.float64))
                t_b_dim += tf.size(b_t, out_type=tf.float64)
                t_b_sum += tf.reduce_sum(tf.cast(b_t,dtype=tf.float64))
                    
            # calculate sparsity levels for unary and binary operators
            sparsity_unary = num_unary_masks/num_unary_weights
            sparsity_binary = num_binary_masks/num_binary_weights
            weight_unary_mean = weight_unary_mean/num_unary_weights
            weight_binary_mean = weight_binary_mean/num_binary_weights
            
            threshold_unary_mean = t_u_sum/t_u_dim
            threshold_binary_mean = t_b_sum/t_b_dim
            
            # sparsity regularization terms
            threshold_input_reg_loss = regression_loss*tf.exp(-threshold_input_mean)
            threshold_model_reg_loss = regression_loss*sum_exp_t/num_model_weights
            threshold_unary_reg_loss = regression_loss*tf.exp(-threshold_unary_mean)
            threshold_binary_reg_loss = regression_loss*tf.exp(-threshold_binary_mean)
            # additional decay factor
            def reg(s, s_t, d):
                return tf.exp(-(s_t/(s_t-tf.minimum(s, s_t)))**d+1.)
            threshold_input_reg_loss *= reg(sparsity_input, self.alpha_sparsity_input, 0.01)
            threshold_model_reg_loss *= reg(sparsity_model, self.alpha_sparsity_model, 0.01)
            threshold_unary_reg_loss *= reg(sparsity_unary, self.alpha_sparsity_unary, 0.01)
            threshold_binary_reg_loss *= reg(sparsity_binary, self.alpha_sparsity_binary, 0.01)
            
            # total loss
            total_loss = regression_loss + (threshold_model_reg_loss + 
                                            threshold_input_reg_loss + 
                                            threshold_unary_reg_loss +
                                            threshold_binary_reg_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                
        self.total_loss_tracker.update_state(total_loss)
        self.regression_loss_tracker.update_state(regression_loss)
        self.threshold_input_reg_loss_tracker.update_state(threshold_input_reg_loss)
        self.threshold_input_mean_tracker.update_state(threshold_input_mean)
        self.threshold_model_reg_loss_tracker.update_state(threshold_model_reg_loss)
        self.threshold_model_mean_tracker.update_state(threshold_model_mean)
        self.threshold_unary_reg_loss_tracker.update_state(threshold_unary_reg_loss)
        self.threshold_unary_mean_tracker.update_state(threshold_unary_mean)
        self.threshold_binary_reg_loss_tracker.update_state(threshold_binary_reg_loss)
        self.threshold_binary_mean_tracker.update_state(threshold_binary_mean)
        self.weight_model_mean_tracker.update_state(weight_model_mean)
        self.weight_input_mean_tracker.update_state(weight_input_mean)
        self.weight_unary_mean_tracker.update_state(weight_unary_mean)
        self.weight_binary_mean_tracker.update_state(weight_binary_mean)
        self.accuracy_tracker.update_state(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
        self.sparsity_input_tracker.update_state(sparsity_input)
        self.sparsity_model_tracker.update_state(sparsity_model)
        self.sparsity_unary_tracker.update_state(sparsity_unary)
        self.sparsity_binary_tracker.update_state(sparsity_binary)
        return {
            'loss': self.total_loss_tracker.result(),
            'regression_loss': self.regression_loss_tracker.result(),
            #'threshold_input_reg_loss': self.threshold_input_reg_loss_tracker.result(),
            #'threshold_input_mean': self.threshold_input_mean_tracker.result(),
            #'threshold_model_reg_loss': self.threshold_model_reg_loss_tracker.result(),
            #'threshold_model_mean': self.threshold_model_mean_tracker.result(),
            #'threshold_unary_reg_loss': self.threshold_unary_reg_loss_tracker.result(),
            #'threshold_unary_mean': self.threshold_unary_mean_tracker.result(),
            #'threshold_binary_reg_loss': self.threshold_binary_reg_loss_tracker.result(),
            #'threshold_binary_mean': self.threshold_binary_mean_tracker.result(),
            #'weight_model_mean': self.weight_model_mean_tracker.result(),
            #'weight_input_mean': self.weight_input_mean_tracker.result(),
            #'weight_unary_mean': self.weight_unary_mean_tracker.result(),
            #'weight_binary_mean': self.weight_binary_mean_tracker.result(),
            'sparsity_input': self.sparsity_input_tracker.result(),
            'sparsity_model': self.sparsity_model_tracker.result(),
            'sparsity_unary': self.sparsity_unary_tracker.result(),
            'sparsity_binary': self.sparsity_binary_tracker.result(),
            'accuracy': self.accuracy_tracker.result(),
        }
    


# set significant digits for expression display
significant_digits = 2

# use sympy to expand the trained model
def get_expressions(neuralSR):
    model_dim, operators, num_operators = neuralSR.get_hyperparameters()
    input_dim, num_hidden_layers, output_dim = model_dim
    
    x=[]
    for i in range(input_dim):
        x.append(sympy.Symbol('x{}'.format(i)))
    x_masked = sympy.Matrix([x])
    
    w_input = neuralSR.model.layers[1].get_weights()[1]
    t_input = neuralSR.model.layers[1].get_weights()[0]
    num_input_masks = tf.reduce_sum(tf.cast(tf.where(w_input-t_input>0., 0., 1.),dtype=tf.float64))
    num_input_weights = tf.size(w_input, out_type=tf.float64)
    
    sparsity_input = num_input_masks/num_input_weights
    
    w_input_masked = sympy.Matrix(tf.where(w_input-t_input>0., w_input, 0.))
    x_print = np.multiply(sympy.Transpose(w_input_masked), x_masked)
    x_masked = np.multiply(sympy.Transpose(w_input_masked), x_masked)

    print('Remaining Inputs after pruning: {}\n'.format(str(x_masked).replace('1.0*','')))
    
    num_masks = 0.
    num_weights = 0.
    
    num_unary_masks = 0.
    num_binary_masks = 0.
    num_unary = 0.
    num_binary = 0.
    
    for i in range(num_hidden_layers+1):
        w = neuralSR.model.layers[i+2].get_weights()[0]
        b = neuralSR.model.layers[i+2].get_weights()[1]
        w_t = neuralSR.model.layers[i+2].get_weights()[2]
        b_t = neuralSR.model.layers[i+2].get_weights()[3]
        
        num_masks += tf.reduce_sum(tf.cast(tf.where(tf.abs(w) - w_t > 0., 0., 1.),dtype=tf.float64))
        num_masks += tf.reduce_sum(tf.cast(tf.where(tf.abs(b) - b_t > 0., 0., 1.),dtype=tf.float64))
        num_weights += tf.size(w, out_type=tf.float64)
        num_weights += tf.size(b, out_type=tf.float64)
        
        w_masked = sympy.Matrix(tf.where(tf.abs(w) - w_t > 0., w, 0.))
        b_masked = sympy.Transpose(sympy.Matrix(tf.where(tf.abs(b) - b_t > 0., b, 0.)))
        
        x_masked = (x_masked * w_masked + b_masked).evalf(significant_digits)
        if i < num_hidden_layers:
            unary = neuralSR.model.layers[i+2].get_weights()[6]
            unary_t = neuralSR.model.layers[i+2].get_weights()[4]
            binary = neuralSR.model.layers[i+2].get_weights()[7]
            binary_t = neuralSR.model.layers[i+2].get_weights()[5]
            num_unary_masks += tf.reduce_sum(tf.cast(tf.where(unary - unary_t > 0., 0., 1.),dtype=tf.float64))
            num_binary_masks += tf.reduce_sum(tf.cast(tf.where(binary - binary_t > 0., 0., 1.),dtype=tf.float64))
            num_unary += tf.size(unary, out_type=tf.float64)
            num_binary += tf.size(binary, out_type=tf.float64)
        elif i == num_hidden_layers:
            num_operators.append([output_dim, 0])
            operators.append([['identity'], [None]])
            
        y_masked = sympy.zeros(1, num_operators[i][0] + num_operators[i][1])
        
        unary_mask = sympy.Matrix(tf.where(unary - unary_t > 0., 1., 0.))
        binary_mask = sympy.Matrix(tf.where(binary - binary_t > 0., 1., 0.))
        for j in range(num_operators[i][0]):
            idx = np.mod(j, len(operators[i][0]))
            if i < num_hidden_layers:
                y_masked[0,j] = (unary_mask[j] * math_operation('sympy',operators[i][0][idx],x_masked[0,j]) +
                             (1.0 - unary_mask[j]) * math_operation('sympy','identity',x_masked[0,j]))
            elif i == num_hidden_layers:
                y_masked[0,j] = x_masked[0,j]
        for j in range(num_operators[i][1]):
            idx = np.mod(j, len(operators[i][1]))
            y_masked[0,num_operators[i][0]+j] = (binary_mask[j] * math_operation('sympy',operators[i][1][idx],x_masked[0,num_operators[i][0]+2*j],x_masked[0,num_operators[i][0]+2*j+1]) +
                                              (1.0 - binary_mask[j]) * math_operation('sympy','+',x_masked[0,num_operators[i][0]+2*j],x_masked[0,num_operators[i][0]+2*j+1]))
        x_masked = y_masked.evalf(significant_digits)
    
    sparsity_model = num_masks/num_weights
    sparsity_unary = num_unary_masks/num_unary
    sparsity_binary = num_binary_masks/num_binary
    
    complexity = []
    for j in range(len(x_masked)):
        c = 0
        for tree_node in sympy.preorder_traversal(x_masked[j]):
            c += 1
        complexity.append(c)
    
    return x_masked, complexity, sparsity_input, sparsity_model, sparsity_unary, sparsity_binary
