
®
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8À

Readout_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameReadout_layer/kernel
}
(Readout_layer/kernel/Read/ReadVariableOpReadVariableOpReadout_layer/kernel*
_output_shapes

:*
dtype0
|
Readout_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameReadout_layer/bias
u
&Readout_layer/bias/Read/ReadVariableOpReadVariableOpReadout_layer/bias*
_output_shapes
:*
dtype0

Dense_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameDense_layer_0/kernel
}
(Dense_layer_0/kernel/Read/ReadVariableOpReadVariableOpDense_layer_0/kernel*
_output_shapes

:*
dtype0
|
Dense_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameDense_layer_0/bias
u
&Dense_layer_0/bias/Read/ReadVariableOpReadVariableOpDense_layer_0/bias*
_output_shapes
:*
dtype0

Dense_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameDense_layer_1/kernel
}
(Dense_layer_1/kernel/Read/ReadVariableOpReadVariableOpDense_layer_1/kernel*
_output_shapes

:*
dtype0
|
Dense_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameDense_layer_1/bias
u
&Dense_layer_1/bias/Read/ReadVariableOpReadVariableOpDense_layer_1/bias*
_output_shapes
:*
dtype0

Dense_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameDense_layer_2/kernel
}
(Dense_layer_2/kernel/Read/ReadVariableOpReadVariableOpDense_layer_2/kernel*
_output_shapes

:*
dtype0
|
Dense_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameDense_layer_2/bias
u
&Dense_layer_2/bias/Read/ReadVariableOpReadVariableOpDense_layer_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ø
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³
value©B¦ B

dense_layers
readout_layer
	variables
regularization_losses
trainable_variables
	keras_api

signatures

0
	1

2
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
­
metrics
	variables
non_trainable_variables
regularization_losses
layer_metrics
trainable_variables

layers
layer_regularization_losses
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

kernel
bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
YW
VARIABLE_VALUEReadout_layer/kernel/readout_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEReadout_layer/bias-readout_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
(metrics
	variables
)non_trainable_variables
regularization_losses
*layer_metrics
trainable_variables

+layers
,layer_regularization_losses
PN
VARIABLE_VALUEDense_layer_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEDense_layer_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEDense_layer_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEDense_layer_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEDense_layer_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEDense_layer_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
	1

2
3
 

0
1
 

0
1
­
-metrics
	variables
.non_trainable_variables
regularization_losses
/layer_metrics
trainable_variables

0layers
1layer_regularization_losses

0
1
 

0
1
­
2metrics
 	variables
3non_trainable_variables
!regularization_losses
4layer_metrics
"trainable_variables

5layers
6layer_regularization_losses

0
1
 

0
1
­
7metrics
$	variables
8non_trainable_variables
%regularization_losses
9layer_metrics
&trainable_variables

:layers
;layer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ë
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Dense_layer_0/kernelDense_layer_0/biasDense_layer_1/kernelDense_layer_1/biasDense_layer_2/kernelDense_layer_2/biasReadout_layer/kernelReadout_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1374053
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(Readout_layer/kernel/Read/ReadVariableOp&Readout_layer/bias/Read/ReadVariableOp(Dense_layer_0/kernel/Read/ReadVariableOp&Dense_layer_0/bias/Read/ReadVariableOp(Dense_layer_1/kernel/Read/ReadVariableOp&Dense_layer_1/bias/Read/ReadVariableOp(Dense_layer_2/kernel/Read/ReadVariableOp&Dense_layer_2/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1374241
Ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameReadout_layer/kernelReadout_layer/biasDense_layer_0/kernelDense_layer_0/biasDense_layer_1/kernelDense_layer_1/biasDense_layer_2/kernelDense_layer_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1374275ÌÜ
ó)

__inference_call_1373877
input_state0
,dense_layer_0_matmul_readvariableop_resource1
-dense_layer_0_biasadd_readvariableop_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,readout_layer_matmul_readvariableop_resource1
-readout_layer_biasadd_readvariableop_resource
identity¢$Dense_layer_0/BiasAdd/ReadVariableOp¢#Dense_layer_0/MatMul/ReadVariableOp¢$Dense_layer_1/BiasAdd/ReadVariableOp¢#Dense_layer_1/MatMul/ReadVariableOp¢$Dense_layer_2/BiasAdd/ReadVariableOp¢#Dense_layer_2/MatMul/ReadVariableOp¢$Readout_layer/BiasAdd/ReadVariableOp¢#Readout_layer/MatMul/ReadVariableOp·
#Dense_layer_0/MatMul/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_0/MatMul/ReadVariableOp¢
Dense_layer_0/MatMulMatMulinput_state+Dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/MatMul¶
$Dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_0/BiasAdd/ReadVariableOp¹
Dense_layer_0/BiasAddBiasAddDense_layer_0/MatMul:product:0,Dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/BiasAdd
Dense_layer_0/ReluReluDense_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/Relu·
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOp·
Dense_layer_1/MatMulMatMul Dense_layer_0/Relu:activations:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/MatMul¶
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOp¹
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/BiasAdd
Dense_layer_1/ReluReluDense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/Relu·
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOp·
Dense_layer_2/MatMulMatMul Dense_layer_1/Relu:activations:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/MatMul¶
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOp¹
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/BiasAdd
Dense_layer_2/ReluReluDense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Relu·
#Readout_layer/MatMul/ReadVariableOpReadVariableOp,readout_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Readout_layer/MatMul/ReadVariableOp·
Readout_layer/MatMulMatMul Dense_layer_2/Relu:activations:0+Readout_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Readout_layer/MatMul¶
$Readout_layer/BiasAdd/ReadVariableOpReadVariableOp-readout_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Readout_layer/BiasAdd/ReadVariableOp¹
Readout_layer/BiasAddBiasAddReadout_layer/MatMul:product:0,Readout_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Readout_layer/BiasAdd¦
IdentityIdentityReadout_layer/BiasAdd:output:0%^Dense_layer_0/BiasAdd/ReadVariableOp$^Dense_layer_0/MatMul/ReadVariableOp%^Dense_layer_1/BiasAdd/ReadVariableOp$^Dense_layer_1/MatMul/ReadVariableOp%^Dense_layer_2/BiasAdd/ReadVariableOp$^Dense_layer_2/MatMul/ReadVariableOp%^Readout_layer/BiasAdd/ReadVariableOp$^Readout_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2L
$Dense_layer_0/BiasAdd/ReadVariableOp$Dense_layer_0/BiasAdd/ReadVariableOp2J
#Dense_layer_0/MatMul/ReadVariableOp#Dense_layer_0/MatMul/ReadVariableOp2L
$Dense_layer_1/BiasAdd/ReadVariableOp$Dense_layer_1/BiasAdd/ReadVariableOp2J
#Dense_layer_1/MatMul/ReadVariableOp#Dense_layer_1/MatMul/ReadVariableOp2L
$Dense_layer_2/BiasAdd/ReadVariableOp$Dense_layer_2/BiasAdd/ReadVariableOp2J
#Dense_layer_2/MatMul/ReadVariableOp#Dense_layer_2/MatMul/ReadVariableOp2L
$Readout_layer/BiasAdd/ReadVariableOp$Readout_layer/BiasAdd/ReadVariableOp2J
#Readout_layer/MatMul/ReadVariableOp#Readout_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
é

/__inference_Dense_layer_1_layer_call_fn_1374174

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_13739382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
Þ
F__inference_q_net_200_layer_call_and_return_conditional_losses_1374008
input_1
dense_layer_0_1373922
dense_layer_0_1373924
dense_layer_1_1373949
dense_layer_1_1373951
dense_layer_2_1373976
dense_layer_2_1373978
readout_layer_1374002
readout_layer_1374004
identity¢%Dense_layer_0/StatefulPartitionedCall¢%Dense_layer_1/StatefulPartitionedCall¢%Dense_layer_2/StatefulPartitionedCall¢%Readout_layer/StatefulPartitionedCall±
%Dense_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_layer_0_1373922dense_layer_0_1373924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_13739112'
%Dense_layer_0/StatefulPartitionedCallØ
%Dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_0/StatefulPartitionedCall:output:0dense_layer_1_1373949dense_layer_1_1373951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_13739382'
%Dense_layer_1/StatefulPartitionedCallØ
%Dense_layer_2/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_1/StatefulPartitionedCall:output:0dense_layer_2_1373976dense_layer_2_1373978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_13739652'
%Dense_layer_2/StatefulPartitionedCallØ
%Readout_layer/StatefulPartitionedCallStatefulPartitionedCall.Dense_layer_2/StatefulPartitionedCall:output:0readout_layer_1374002readout_layer_1374004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Readout_layer_layer_call_and_return_conditional_losses_13739912'
%Readout_layer/StatefulPartitionedCall¢
IdentityIdentity.Readout_layer/StatefulPartitionedCall:output:0&^Dense_layer_0/StatefulPartitionedCall&^Dense_layer_1/StatefulPartitionedCall&^Dense_layer_2/StatefulPartitionedCall&^Readout_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2N
%Dense_layer_0/StatefulPartitionedCall%Dense_layer_0/StatefulPartitionedCall2N
%Dense_layer_1/StatefulPartitionedCall%Dense_layer_1/StatefulPartitionedCall2N
%Dense_layer_2/StatefulPartitionedCall%Dense_layer_2/StatefulPartitionedCall2N
%Readout_layer/StatefulPartitionedCall%Readout_layer/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ú
Õ
%__inference_signature_wrapper_1374053
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_13738962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
·	

"__inference__wrapped_model_1373896
input_1
q_net_200_1373878
q_net_200_1373880
q_net_200_1373882
q_net_200_1373884
q_net_200_1373886
q_net_200_1373888
q_net_200_1373890
q_net_200_1373892
identity¢!q_net_200/StatefulPartitionedCallí
!q_net_200/StatefulPartitionedCallStatefulPartitionedCallinput_1q_net_200_1373878q_net_200_1373880q_net_200_1373882q_net_200_1373884q_net_200_1373886q_net_200_1373888q_net_200_1373890q_net_200_1373892*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_13738772#
!q_net_200/StatefulPartitionedCall¢
IdentityIdentity*q_net_200/StatefulPartitionedCall:output:0"^q_net_200/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2F
!q_net_200/StatefulPartitionedCall!q_net_200/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
ã
J__inference_Readout_layer_layer_call_and_return_conditional_losses_1373991

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
Û
+__inference_q_net_200_layer_call_fn_1374030
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_q_net_200_layer_call_and_return_conditional_losses_13740082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
	
ã
J__inference_Readout_layer_layer_call_and_return_conditional_losses_1374125

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
ã
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_1373965

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

/__inference_Dense_layer_0_layer_call_fn_1374154

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_13739112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó)

__inference_call_1374115
input_state0
,dense_layer_0_matmul_readvariableop_resource1
-dense_layer_0_biasadd_readvariableop_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,readout_layer_matmul_readvariableop_resource1
-readout_layer_biasadd_readvariableop_resource
identity¢$Dense_layer_0/BiasAdd/ReadVariableOp¢#Dense_layer_0/MatMul/ReadVariableOp¢$Dense_layer_1/BiasAdd/ReadVariableOp¢#Dense_layer_1/MatMul/ReadVariableOp¢$Dense_layer_2/BiasAdd/ReadVariableOp¢#Dense_layer_2/MatMul/ReadVariableOp¢$Readout_layer/BiasAdd/ReadVariableOp¢#Readout_layer/MatMul/ReadVariableOp·
#Dense_layer_0/MatMul/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_0/MatMul/ReadVariableOp¢
Dense_layer_0/MatMulMatMulinput_state+Dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/MatMul¶
$Dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_0/BiasAdd/ReadVariableOp¹
Dense_layer_0/BiasAddBiasAddDense_layer_0/MatMul:product:0,Dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/BiasAdd
Dense_layer_0/ReluReluDense_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_0/Relu·
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOp·
Dense_layer_1/MatMulMatMul Dense_layer_0/Relu:activations:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/MatMul¶
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOp¹
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/BiasAdd
Dense_layer_1/ReluReluDense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_1/Relu·
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOp·
Dense_layer_2/MatMulMatMul Dense_layer_1/Relu:activations:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/MatMul¶
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOp¹
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/BiasAdd
Dense_layer_2/ReluReluDense_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Dense_layer_2/Relu·
#Readout_layer/MatMul/ReadVariableOpReadVariableOp,readout_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Readout_layer/MatMul/ReadVariableOp·
Readout_layer/MatMulMatMul Dense_layer_2/Relu:activations:0+Readout_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Readout_layer/MatMul¶
$Readout_layer/BiasAdd/ReadVariableOpReadVariableOp-readout_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Readout_layer/BiasAdd/ReadVariableOp¹
Readout_layer/BiasAddBiasAddReadout_layer/MatMul:product:0,Readout_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Readout_layer/BiasAdd¦
IdentityIdentityReadout_layer/BiasAdd:output:0%^Dense_layer_0/BiasAdd/ReadVariableOp$^Dense_layer_0/MatMul/ReadVariableOp%^Dense_layer_1/BiasAdd/ReadVariableOp$^Dense_layer_1/MatMul/ReadVariableOp%^Dense_layer_2/BiasAdd/ReadVariableOp$^Dense_layer_2/MatMul/ReadVariableOp%^Readout_layer/BiasAdd/ReadVariableOp$^Readout_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2L
$Dense_layer_0/BiasAdd/ReadVariableOp$Dense_layer_0/BiasAdd/ReadVariableOp2J
#Dense_layer_0/MatMul/ReadVariableOp#Dense_layer_0/MatMul/ReadVariableOp2L
$Dense_layer_1/BiasAdd/ReadVariableOp$Dense_layer_1/BiasAdd/ReadVariableOp2J
#Dense_layer_1/MatMul/ReadVariableOp#Dense_layer_1/MatMul/ReadVariableOp2L
$Dense_layer_2/BiasAdd/ReadVariableOp$Dense_layer_2/BiasAdd/ReadVariableOp2J
#Dense_layer_2/MatMul/ReadVariableOp#Dense_layer_2/MatMul/ReadVariableOp2L
$Readout_layer/BiasAdd/ReadVariableOp$Readout_layer/BiasAdd/ReadVariableOp2J
#Readout_layer/MatMul/ReadVariableOp#Readout_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
ô	
ã
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_1374165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
ã
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_1373911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

/__inference_Dense_layer_2_layer_call_fn_1374194

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_13739652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
ã
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_1374185

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò(

__inference_call_1374084
input_state0
,dense_layer_0_matmul_readvariableop_resource1
-dense_layer_0_biasadd_readvariableop_resource0
,dense_layer_1_matmul_readvariableop_resource1
-dense_layer_1_biasadd_readvariableop_resource0
,dense_layer_2_matmul_readvariableop_resource1
-dense_layer_2_biasadd_readvariableop_resource0
,readout_layer_matmul_readvariableop_resource1
-readout_layer_biasadd_readvariableop_resource
identity¢$Dense_layer_0/BiasAdd/ReadVariableOp¢#Dense_layer_0/MatMul/ReadVariableOp¢$Dense_layer_1/BiasAdd/ReadVariableOp¢#Dense_layer_1/MatMul/ReadVariableOp¢$Dense_layer_2/BiasAdd/ReadVariableOp¢#Dense_layer_2/MatMul/ReadVariableOp¢$Readout_layer/BiasAdd/ReadVariableOp¢#Readout_layer/MatMul/ReadVariableOp·
#Dense_layer_0/MatMul/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_0/MatMul/ReadVariableOp
Dense_layer_0/MatMulMatMulinput_state+Dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_0/MatMul¶
$Dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_0/BiasAdd/ReadVariableOp°
Dense_layer_0/BiasAddBiasAddDense_layer_0/MatMul:product:0,Dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_0/BiasAddy
Dense_layer_0/ReluReluDense_layer_0/BiasAdd:output:0*
T0*
_output_shapes

:2
Dense_layer_0/Relu·
#Dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_1/MatMul/ReadVariableOp®
Dense_layer_1/MatMulMatMul Dense_layer_0/Relu:activations:0+Dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_1/MatMul¶
$Dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_1/BiasAdd/ReadVariableOp°
Dense_layer_1/BiasAddBiasAddDense_layer_1/MatMul:product:0,Dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_1/BiasAddy
Dense_layer_1/ReluReluDense_layer_1/BiasAdd:output:0*
T0*
_output_shapes

:2
Dense_layer_1/Relu·
#Dense_layer_2/MatMul/ReadVariableOpReadVariableOp,dense_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Dense_layer_2/MatMul/ReadVariableOp®
Dense_layer_2/MatMulMatMul Dense_layer_1/Relu:activations:0+Dense_layer_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_2/MatMul¶
$Dense_layer_2/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Dense_layer_2/BiasAdd/ReadVariableOp°
Dense_layer_2/BiasAddBiasAddDense_layer_2/MatMul:product:0,Dense_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Dense_layer_2/BiasAddy
Dense_layer_2/ReluReluDense_layer_2/BiasAdd:output:0*
T0*
_output_shapes

:2
Dense_layer_2/Relu·
#Readout_layer/MatMul/ReadVariableOpReadVariableOp,readout_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#Readout_layer/MatMul/ReadVariableOp®
Readout_layer/MatMulMatMul Dense_layer_2/Relu:activations:0+Readout_layer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Readout_layer/MatMul¶
$Readout_layer/BiasAdd/ReadVariableOpReadVariableOp-readout_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Readout_layer/BiasAdd/ReadVariableOp°
Readout_layer/BiasAddBiasAddReadout_layer/MatMul:product:0,Readout_layer/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Readout_layer/BiasAdd
IdentityIdentityReadout_layer/BiasAdd:output:0%^Dense_layer_0/BiasAdd/ReadVariableOp$^Dense_layer_0/MatMul/ReadVariableOp%^Dense_layer_1/BiasAdd/ReadVariableOp$^Dense_layer_1/MatMul/ReadVariableOp%^Dense_layer_2/BiasAdd/ReadVariableOp$^Dense_layer_2/MatMul/ReadVariableOp%^Readout_layer/BiasAdd/ReadVariableOp$^Readout_layer/MatMul/ReadVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*=
_input_shapes,
*:::::::::2L
$Dense_layer_0/BiasAdd/ReadVariableOp$Dense_layer_0/BiasAdd/ReadVariableOp2J
#Dense_layer_0/MatMul/ReadVariableOp#Dense_layer_0/MatMul/ReadVariableOp2L
$Dense_layer_1/BiasAdd/ReadVariableOp$Dense_layer_1/BiasAdd/ReadVariableOp2J
#Dense_layer_1/MatMul/ReadVariableOp#Dense_layer_1/MatMul/ReadVariableOp2L
$Dense_layer_2/BiasAdd/ReadVariableOp$Dense_layer_2/BiasAdd/ReadVariableOp2J
#Dense_layer_2/MatMul/ReadVariableOp#Dense_layer_2/MatMul/ReadVariableOp2L
$Readout_layer/BiasAdd/ReadVariableOp$Readout_layer/BiasAdd/ReadVariableOp2J
#Readout_layer/MatMul/ReadVariableOp#Readout_layer/MatMul/ReadVariableOp:K G

_output_shapes

:
%
_user_specified_nameinput_state
Î%
Í
#__inference__traced_restore_1374275
file_prefix)
%assignvariableop_readout_layer_kernel)
%assignvariableop_1_readout_layer_bias+
'assignvariableop_2_dense_layer_0_kernel)
%assignvariableop_3_dense_layer_0_bias+
'assignvariableop_4_dense_layer_1_kernel)
%assignvariableop_5_dense_layer_1_bias+
'assignvariableop_6_dense_layer_2_kernel)
%assignvariableop_7_dense_layer_2_bias

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7÷
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*
valueùBö	B/readout_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¤
AssignVariableOpAssignVariableOp%assignvariableop_readout_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ª
AssignVariableOp_1AssignVariableOp%assignvariableop_1_readout_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¬
AssignVariableOp_2AssignVariableOp'assignvariableop_2_dense_layer_0_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ª
AssignVariableOp_3AssignVariableOp%assignvariableop_3_dense_layer_0_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_dense_layer_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_dense_layer_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¬
AssignVariableOp_6AssignVariableOp'assignvariableop_6_dense_layer_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ª
AssignVariableOp_7AssignVariableOp%assignvariableop_7_dense_layer_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ô	
ã
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_1373938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô	
ã
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_1374145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

/__inference_Readout_layer_layer_call_fn_1374134

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_Readout_layer_layer_call_and_return_conditional_losses_13739912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

 __inference__traced_save_1374241
file_prefix3
/savev2_readout_layer_kernel_read_readvariableop1
-savev2_readout_layer_bias_read_readvariableop3
/savev2_dense_layer_0_kernel_read_readvariableop1
-savev2_dense_layer_0_bias_read_readvariableop3
/savev2_dense_layer_1_kernel_read_readvariableop1
-savev2_dense_layer_1_bias_read_readvariableop3
/savev2_dense_layer_2_kernel_read_readvariableop1
-savev2_dense_layer_2_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameñ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*
valueùBö	B/readout_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesÂ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_readout_layer_kernel_read_readvariableop-savev2_readout_layer_bias_read_readvariableop/savev2_dense_layer_0_kernel_read_readvariableop-savev2_dense_layer_0_bias_read_readvariableop/savev2_dense_layer_1_kernel_read_readvariableop-savev2_dense_layer_1_bias_read_readvariableop/savev2_dense_layer_2_kernel_read_readvariableop-savev2_dense_layer_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
q_values0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ái
Þ
dense_layers
readout_layer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses
?call"ó
_tf_keras_modelÙ{"class_name": "QNet", "name": "q_net_200", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "QNet"}}
5
0
	1

2"
trackable_list_wrapper
ú

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "Readout_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Readout_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ê
metrics
	variables
non_trainable_variables
regularization_losses
layer_metrics
trainable_variables

layers
layer_regularization_losses
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Bserving_default"
signature_map
÷

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "Dense_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_layer_0", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
ù

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
E__call__
*F&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Dense_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_layer_1", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
ù

kernel
bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
G__call__
*H&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Dense_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_layer_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16]}}
&:$2Readout_layer/kernel
 :2Readout_layer/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
(metrics
	variables
)non_trainable_variables
regularization_losses
*layer_metrics
trainable_variables

+layers
,layer_regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
&:$2Dense_layer_0/kernel
 :2Dense_layer_0/bias
&:$2Dense_layer_1/kernel
 :2Dense_layer_1/bias
&:$2Dense_layer_2/kernel
 :2Dense_layer_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
-metrics
	variables
.non_trainable_variables
regularization_losses
/layer_metrics
trainable_variables

0layers
1layer_regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
2metrics
 	variables
3non_trainable_variables
!regularization_losses
4layer_metrics
"trainable_variables

5layers
6layer_regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7metrics
$	variables
8non_trainable_variables
%regularization_losses
9layer_metrics
&trainable_variables

:layers
;layer_regularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
þ2û
+__inference_q_net_200_layer_call_fn_1374030Ë
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
à2Ý
"__inference__wrapped_model_1373896¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2
F__inference_q_net_200_layer_call_and_return_conditional_losses_1374008Ë
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
á2Þ
__inference_call_1374084
__inference_call_1374115§
²
FullArgSpec"
args
jself
jinput_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_Readout_layer_layer_call_fn_1374134¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_Readout_layer_layer_call_and_return_conditional_losses_1374125¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
%__inference_signature_wrapper_1374053input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_Dense_layer_0_layer_call_fn_1374154¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_1374145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_Dense_layer_1_layer_call_fn_1374174¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_1374165¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_Dense_layer_2_layer_call_fn_1374194¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_1374185¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ª
J__inference_Dense_layer_0_layer_call_and_return_conditional_losses_1374145\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Dense_layer_0_layer_call_fn_1374154O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
J__inference_Dense_layer_1_layer_call_and_return_conditional_losses_1374165\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Dense_layer_1_layer_call_fn_1374174O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
J__inference_Dense_layer_2_layer_call_and_return_conditional_losses_1374185\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Dense_layer_2_layer_call_fn_1374194O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
J__inference_Readout_layer_layer_call_and_return_conditional_losses_1374125\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_Readout_layer_layer_call_fn_1374134O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
"__inference__wrapped_model_1373896q0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
q_values"
q_valuesÿÿÿÿÿÿÿÿÿ
__inference_call_1374084c+¢(
!¢

input_state
ª "*ª'
%
q_values
q_values
__inference_call_1374115u4¢1
*¢'
%"
input_stateÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
q_values"
q_valuesÿÿÿÿÿÿÿÿÿÇ
F__inference_q_net_200_layer_call_and_return_conditional_losses_1374008}0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "?¢<
5ª2
0
q_values$!

0/q_valuesÿÿÿÿÿÿÿÿÿ
  
+__inference_q_net_200_layer_call_fn_1374030q0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
q_values"
q_valuesÿÿÿÿÿÿÿÿÿ¥
%__inference_signature_wrapper_1374053|;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
q_values"
q_valuesÿÿÿÿÿÿÿÿÿ