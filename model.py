from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import *
import os 

################Unet structure###########################
def Unet_identity(layer,channel,index):
    layer = Conv2D(channel , (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = 'unet'+index.pop(0))(layer)
    layer = Conv2D(channel , (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = 'unet'+index.pop(0))(layer)
    return layer

def Unet_concat_identity(layers,channel,index):
    merge = concatenate(layers,axis=3)
    layer = Unet_identity(merge,channel,index)
    return layer 

def Unet(node, bridge):

    index = list([str(i) for i in range(1,101)])

    conv1 = Unet_identity(node, 64, index)
    down1 = MaxPool2D(pool_size = (2,2))(conv1)
    
    conv2 = Unet_identity(down1, 128, index)
    down2 = MaxPool2D(pool_size = (2,2))(conv2)
    
    conv3 = Unet_identity(down2, 256, index)
    down3 = MaxPool2D(pool_size = (2,2))(conv3)

    conv4 = Unet_identity(down3, 512, index)
    down4 = MaxPool2D(pool_size = (2,2))(conv4)
    
    conv5 = Unet_identity(down4, 1024, index)
    up5 = UpSampling2D(size = (2,2))(conv5)

    conv4 = Unet_concat_identity([conv4,up5],512, index)
    up4 = UpSampling2D(size = (2,2))(conv4)

    conv3 = Unet_concat_identity([conv3,up4],256, index)
    up3 = UpSampling2D(size = (2,2))(conv3)

    conv2 = Unet_concat_identity([conv2,up3],128, index)
    up2 = UpSampling2D(size = (2,2))(conv2)

    conv1 = Unet_concat_identity([conv1,up2],64, index)
    unet_out = Conv2D(1, (1,1), kernel_initializer = 'he_normal',activation = 'sigmoid',name = 'unet_out')(conv1)

    if bridge:
        pixel_w_conv4 = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer = 'he_normal',name = 'bridge_weight4')(conv5)
        pixel_w_conv3 = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer = 'he_normal',name = 'bridge_weight3')(conv4)
        pixel_w_conv2 = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer = 'he_normal',name = 'bridge_weight2')(conv3)
        pixel_w_conv1 = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer = 'he_normal',name = 'bridge_weight1')(conv2)

        return pixel_w_conv4, pixel_w_conv3, pixel_w_conv2, pixel_w_conv1, unet_out

    return unet_out
#############################################################

###########ResNet structure########################
def res_block(node,filters,first = True,stride = 2):

    filter1, filter2, filter3 = filters
   
    if first:
        x = Conv2D(filter1, (1,1), kernel_initializer = 'he_normal',strides = stride)(node)
    else:
        x = Conv2D(filter1, (1,1),kernel_initializer = 'he_normal')(node)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (3,3), padding='same',kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1,1),kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    
    if first:
        shortcut = Conv2D(filter3, (1,1),strides = stride,kernel_initializer = 'he_normal')(node)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x,shortcut])
    else:
        x = Add()([x, node])
   
    x = Activation('relu')(x)
    return x


def Bridge_Resnet(input_shape = (224,224,3), bridge = True, class_only = False):
    
    input_tensor = Input(shape=input_shape, dtype='float32')
    if class_only:
        bridge = False

    #Unet 
    if bridge:
        w4, w3, w2, w1, unet_out = Unet(input_tensor, bridge)
    else:
        unet_out = Unet(input_tensor, bridge)
    
    #Stage 1
    block1 = ZeroPadding2D(3)(input_tensor)
    block1 = Conv2D(64, (7,7), strides=2,kernel_initializer = 'he_normal')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    #Stage 1 - bridge connection
    if bridge:
        block1 = Multiply()([w1,block1])
    
    #Stage 2
    block2 = MaxPooling2D((3,3), strides=2, padding = 'same')(block1)
    block2 = res_block(block2, [64,64,256], first=True, stride=1)
    for _ in range(2):
        block2 = res_block(block2, [64,64,256],first=False)
    #Stage 2 - bridge connection
    if bridge:
        block2 = Multiply()([w2,block2])

    #Stage 3
    block3 = res_block(block2, [128,128,512], first=True)
    for _ in range(3):
        block3 = res_block(block3, [128,128,512],first=False)
    #Stage 3 - bridge connection
    if bridge:
        block3 = Multiply()([w3,block3])

    #Stage 4
    block4 = res_block(block3, [256,256,1024], first=True)
    for _ in range(5):
        block4 = res_block(block4, [256,256,1024],first=False)
    #Stage 4 - bridge connection
    if bridge:
        block4 = Multiply()([w4,block4])

    #Stage 5
    block5 = res_block(block4, [512,512,2048], first=True)
    for _ in range(2):
        block5 = res_block(block5, [512,512,2048],first=False)

    #FC
    flat = GlobalAveragePooling2D()(block5)
    resnet_out = Dense(1, activation = 'relu', kernel_initializer = 'he_normal',name = 'class_out')(flat)
    
    if class_only:
        return Model(inputs = input_tensor, outputs = resnet_out)

    return Model(inputs = input_tensor, outputs = [unet_out,resnet_out])



###Xception##########
def Xception_flow(x, filters, flow, first):
    if flow == 'Entry':
        residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        residual = BatchNormalization()(residual)
        
        if not first:
            x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        return Add()([x,residual])
    
    elif flow == 'Middle':
        residual = x
        
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
         
        return Add()([x, residual])
    
    elif flow == 'Exit':
        filter1, filter2 = filters
        residual = Conv2D(filter2, (1, 1), strides=(2, 2), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        residual = BatchNormalization()(residual)        
        
        x = Activation('relu')(x)
        x = SeparableConv2D(filter1, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filter2, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        return Add()([x, residual])


def Bridge_Xception(input_shape = (224,224,3), bridge= True, class_only = False):
    
    input_tensor = Input(shape=input_shape, dtype='float32')
    if class_only:
        bridge = False

    #Unet 
    if bridge:
        w4, w3, w2, w1, unet_out = Unet(input_tensor, bridge)
    else:
        unet_out = Unet(input_tensor, bridge)
    
    #Entry flow
    Entry = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding = 'same',kernel_initializer = 'he_normal')(input_tensor)
    Entry = BatchNormalization()(Entry)
    Entry = Activation('relu')(Entry)

    Entry = Conv2D(64, (3, 3), use_bias=False, padding = 'same',kernel_initializer = 'he_normal')(Entry)
    Entry = BatchNormalization()(Entry)
    Entry = Activation('relu')(Entry)

    #Entry flow - bridge connection 1 
    if bridge:
        Entry = Multiply()([w1,Entry])
    
    Entry = Xception_flow(Entry, 128, flow = 'Entry', first = True)

    #Entry flow - bridge connection 2
    if bridge:
        Entry = Multiply()([w2,Entry])

    Entry = Xception_flow(Entry, 256, flow = 'Entry', first = False)
    
    #Entry flow - bridge connection 3
    if bridge:
        Entry = Multiply()([w3,Entry])
    
    Entry = Xception_flow(Entry, 728, flow = 'Entry', first = False)

    #Middle flow
    Middle = Xception_flow(Entry, 728, flow = 'Middle', first = True)
    for _ in range(7):
        Middle = Xception_flow(Middle, 728, flow = 'Middle', first = False)
    
     
    #Middle flow - bridge connection
    if bridge:
        Middle = Multiply()([w4,Middle])
    
    #Exit flow
    Exit = Xception_flow(Middle, [728,1024], flow = 'Exit', first = True)

    Exit = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(Exit)
    Exit = BatchNormalization()(Exit)
    Exit = Activation('relu')(Exit)
    
    Exit = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False,kernel_initializer = 'he_normal')(Exit)
    Exit = BatchNormalization()(Exit)
    Exit = Activation('relu')(Exit)
    
    #FC
    fc = GlobalAveragePooling2D()(Exit)
    xception_out = Dense(1, activation='relu', name = 'class_out',kernel_initializer = 'he_normal')(fc)
    
    if class_only:
        return Model(inputs = input_tensor, outputs = xception_out)

    return Model(inputs = input_tensor, outputs = [unet_out, xception_out])
#############################################################
