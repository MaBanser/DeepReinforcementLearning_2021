??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
delete_old_dirsbool(?
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
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
Policy_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namePolicy_readout/kernel
?
)Policy_readout/kernel/Read/ReadVariableOpReadVariableOpPolicy_readout/kernel*
_output_shapes
:	?*
dtype0
~
Policy_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namePolicy_readout/bias
w
'Policy_readout/bias/Read/ReadVariableOpReadVariableOpPolicy_readout/bias*
_output_shapes
:*
dtype0
?
Critic_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameCritic_readout/kernel
?
)Critic_readout/kernel/Read/ReadVariableOpReadVariableOpCritic_readout/kernel*
_output_shapes
:	?*
dtype0
~
Critic_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameCritic_readout/bias
w
'Critic_readout/bias/Read/ReadVariableOpReadVariableOpCritic_readout/bias*
_output_shapes
:*
dtype0
?
readin_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_namereadin_1/kernel
{
#readin_1/kernel/Read/ReadVariableOpReadVariableOpreadin_1/kernel*&
_output_shapes
:@*
dtype0
r
readin_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namereadin_1/bias
k
!readin_1/bias/Read/ReadVariableOpReadVariableOpreadin_1/bias*
_output_shapes
:@*
dtype0
?
readin_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_namereadin_2/kernel
{
#readin_2/kernel/Read/ReadVariableOpReadVariableOpreadin_2/kernel*&
_output_shapes
:@@*
dtype0
r
readin_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namereadin_2/bias
k
!readin_2/bias/Read/ReadVariableOpReadVariableOpreadin_2/bias*
_output_shapes
:@*
dtype0
?
readin_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_namereadin_3/kernel
{
#readin_3/kernel/Read/ReadVariableOpReadVariableOpreadin_3/kernel*&
_output_shapes
:@@*
dtype0
r
readin_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namereadin_3/bias
k
!readin_3/bias/Read/ReadVariableOpReadVariableOpreadin_3/bias*
_output_shapes
:@*
dtype0
?
readin_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_namereadin_4/kernel
{
#readin_4/kernel/Read/ReadVariableOpReadVariableOpreadin_4/kernel*&
_output_shapes
:@@*
dtype0
r
readin_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namereadin_4/bias
k
!readin_4/bias/Read/ReadVariableOpReadVariableOpreadin_4/bias*
_output_shapes
:@*
dtype0
?
Policy_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namePolicy_layer_1/kernel
?
)Policy_layer_1/kernel/Read/ReadVariableOpReadVariableOpPolicy_layer_1/kernel*&
_output_shapes
:@*
dtype0
~
Policy_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namePolicy_layer_1/bias
w
'Policy_layer_1/bias/Read/ReadVariableOpReadVariableOpPolicy_layer_1/bias*
_output_shapes
:*
dtype0
?
Critic_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameCritic_layer_1/kernel
?
)Critic_layer_1/kernel/Read/ReadVariableOpReadVariableOpCritic_layer_1/kernel*&
_output_shapes
:@*
dtype0
~
Critic_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameCritic_layer_1/bias
w
'Critic_layer_1/bias/Read/ReadVariableOpReadVariableOpCritic_layer_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?(
value?(B?( B?(
?
readin_layers
policy_layers
readout_policy
value_layers
readout_value
	variables
regularization_losses
trainable_variables
		keras_api


signatures

0
1
2
3

0
1
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

0
1
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
v
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
10
11
)12
*13
14
15
 
v
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
10
11
)12
*13
14
15
?
	variables
+layer_metrics
regularization_losses
,layer_regularization_losses
trainable_variables
-metrics

.layers
/non_trainable_variables
 
h

kernel
 bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

!kernel
"bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

#kernel
$bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
h

%kernel
&bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

'kernel
(bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
R
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
[Y
VARIABLE_VALUEPolicy_readout/kernel0readout_policy/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEPolicy_readout/bias.readout_policy/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
Hlayer_metrics
regularization_losses
Ilayer_regularization_losses
trainable_variables
Jmetrics

Klayers
Lnon_trainable_variables
h

)kernel
*bias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
R
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
ZX
VARIABLE_VALUECritic_readout/kernel/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUECritic_readout/bias-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
Ulayer_metrics
regularization_losses
Vlayer_regularization_losses
trainable_variables
Wmetrics

Xlayers
Ynon_trainable_variables
KI
VARIABLE_VALUEreadin_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEreadin_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEreadin_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEreadin_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEreadin_3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEreadin_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEreadin_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEreadin_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEPolicy_layer_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEPolicy_layer_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUECritic_layer_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUECritic_layer_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
F
0
1
2
3
4
5
6
7
8
9
 

0
 1
 

0
 1
?
0	variables
Zlayer_metrics
1regularization_losses
[layer_regularization_losses
2trainable_variables
\metrics

]layers
^non_trainable_variables

!0
"1
 

!0
"1
?
4	variables
_layer_metrics
5regularization_losses
`layer_regularization_losses
6trainable_variables
ametrics

blayers
cnon_trainable_variables

#0
$1
 

#0
$1
?
8	variables
dlayer_metrics
9regularization_losses
elayer_regularization_losses
:trainable_variables
fmetrics

glayers
hnon_trainable_variables

%0
&1
 

%0
&1
?
<	variables
ilayer_metrics
=regularization_losses
jlayer_regularization_losses
>trainable_variables
kmetrics

llayers
mnon_trainable_variables

'0
(1
 

'0
(1
?
@	variables
nlayer_metrics
Aregularization_losses
olayer_regularization_losses
Btrainable_variables
pmetrics

qlayers
rnon_trainable_variables
 
 
 
?
D	variables
slayer_metrics
Eregularization_losses
tlayer_regularization_losses
Ftrainable_variables
umetrics

vlayers
wnon_trainable_variables
 
 
 
 
 

)0
*1
 

)0
*1
?
M	variables
xlayer_metrics
Nregularization_losses
ylayer_regularization_losses
Otrainable_variables
zmetrics

{layers
|non_trainable_variables
 
 
 
?
Q	variables
}layer_metrics
Rregularization_losses
~layer_regularization_losses
Strainable_variables
metrics
?layers
?non_trainable_variables
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
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1readin_1/kernelreadin_1/biasreadin_2/kernelreadin_2/biasreadin_3/kernelreadin_3/biasreadin_4/kernelreadin_4/biasPolicy_layer_1/kernelPolicy_layer_1/biasPolicy_readout/kernelPolicy_readout/biasCritic_layer_1/kernelCritic_layer_1/biasCritic_readout/kernelCritic_readout/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_96166205
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Policy_readout/kernel/Read/ReadVariableOp'Policy_readout/bias/Read/ReadVariableOp)Critic_readout/kernel/Read/ReadVariableOp'Critic_readout/bias/Read/ReadVariableOp#readin_1/kernel/Read/ReadVariableOp!readin_1/bias/Read/ReadVariableOp#readin_2/kernel/Read/ReadVariableOp!readin_2/bias/Read/ReadVariableOp#readin_3/kernel/Read/ReadVariableOp!readin_3/bias/Read/ReadVariableOp#readin_4/kernel/Read/ReadVariableOp!readin_4/bias/Read/ReadVariableOp)Policy_layer_1/kernel/Read/ReadVariableOp'Policy_layer_1/bias/Read/ReadVariableOp)Critic_layer_1/kernel/Read/ReadVariableOp'Critic_layer_1/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_96166593
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamePolicy_readout/kernelPolicy_readout/biasCritic_readout/kernelCritic_readout/biasreadin_1/kernelreadin_1/biasreadin_2/kernelreadin_2/biasreadin_3/kernelreadin_3/biasreadin_4/kernelreadin_4/biasPolicy_layer_1/kernelPolicy_layer_1/biasCritic_layer_1/kernelCritic_layer_1/bias*
Tin
2*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_96166651??
?
e
I__inference_flatten_727_layer_call_and_return_conditional_losses_96166086

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_flatten_726_layer_call_and_return_conditional_losses_96166017

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_flatten_726_layer_call_fn_96166490

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flatten_726_layer_call_and_return_conditional_losses_961660172
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_readin_4_layer_call_and_return_conditional_losses_96166450

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_flatten_726_layer_call_and_return_conditional_losses_96166485

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_96166470

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
L__inference_Critic_readout_layer_call_and_return_conditional_losses_96166105

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_readin_2_layer_call_fn_96166419

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_2_layer_call_and_return_conditional_losses_961659142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_flatten_727_layer_call_fn_96166521

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flatten_727_layer_call_and_return_conditional_losses_961660862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?\
?
__inference_call_96166272
observation+
'readin_1_conv2d_readvariableop_resource,
(readin_1_biasadd_readvariableop_resource+
'readin_2_conv2d_readvariableop_resource,
(readin_2_biasadd_readvariableop_resource+
'readin_3_conv2d_readvariableop_resource,
(readin_3_biasadd_readvariableop_resource+
'readin_4_conv2d_readvariableop_resource,
(readin_4_biasadd_readvariableop_resource1
-policy_layer_1_conv2d_readvariableop_resource2
.policy_layer_1_biasadd_readvariableop_resource1
-policy_readout_matmul_readvariableop_resource2
.policy_readout_biasadd_readvariableop_resource1
-critic_layer_1_conv2d_readvariableop_resource2
.critic_layer_1_biasadd_readvariableop_resource1
-critic_readout_matmul_readvariableop_resource2
.critic_readout_biasadd_readvariableop_resource
identity

identity_1??%Critic_layer_1/BiasAdd/ReadVariableOp?$Critic_layer_1/Conv2D/ReadVariableOp?%Critic_readout/BiasAdd/ReadVariableOp?$Critic_readout/MatMul/ReadVariableOp?%Policy_layer_1/BiasAdd/ReadVariableOp?$Policy_layer_1/Conv2D/ReadVariableOp?%Policy_readout/BiasAdd/ReadVariableOp?$Policy_readout/MatMul/ReadVariableOp?readin_1/BiasAdd/ReadVariableOp?readin_1/Conv2D/ReadVariableOp?readin_2/BiasAdd/ReadVariableOp?readin_2/Conv2D/ReadVariableOp?readin_3/BiasAdd/ReadVariableOp?readin_3/Conv2D/ReadVariableOp?readin_4/BiasAdd/ReadVariableOp?readin_4/Conv2D/ReadVariableOp?
readin_1/Conv2D/ReadVariableOpReadVariableOp'readin_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
readin_1/Conv2D/ReadVariableOp?
readin_1/Conv2DConv2Dobservation&readin_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
readin_1/Conv2D?
readin_1/BiasAdd/ReadVariableOpReadVariableOp(readin_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_1/BiasAdd/ReadVariableOp?
readin_1/BiasAddBiasAddreadin_1/Conv2D:output:0'readin_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
readin_1/BiasAddr
readin_1/TanhTanhreadin_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
readin_1/Tanh?
readin_2/Conv2D/ReadVariableOpReadVariableOp'readin_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_2/Conv2D/ReadVariableOp?
readin_2/Conv2DConv2Dreadin_1/Tanh:y:0&readin_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
readin_2/Conv2D?
readin_2/BiasAdd/ReadVariableOpReadVariableOp(readin_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_2/BiasAdd/ReadVariableOp?
readin_2/BiasAddBiasAddreadin_2/Conv2D:output:0'readin_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
readin_2/BiasAddr
readin_2/TanhTanhreadin_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
readin_2/Tanh?
readin_3/Conv2D/ReadVariableOpReadVariableOp'readin_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_3/Conv2D/ReadVariableOp?
readin_3/Conv2DConv2Dreadin_2/Tanh:y:0&readin_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
readin_3/Conv2D?
readin_3/BiasAdd/ReadVariableOpReadVariableOp(readin_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_3/BiasAdd/ReadVariableOp?
readin_3/BiasAddBiasAddreadin_3/Conv2D:output:0'readin_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
readin_3/BiasAddr
readin_3/TanhTanhreadin_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
readin_3/Tanh?
readin_4/Conv2D/ReadVariableOpReadVariableOp'readin_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_4/Conv2D/ReadVariableOp?
readin_4/Conv2DConv2Dreadin_3/Tanh:y:0&readin_4/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
readin_4/Conv2D?
readin_4/BiasAdd/ReadVariableOpReadVariableOp(readin_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_4/BiasAdd/ReadVariableOp?
readin_4/BiasAddBiasAddreadin_4/Conv2D:output:0'readin_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
readin_4/BiasAddr
readin_4/TanhTanhreadin_4/BiasAdd:output:0*
T0*&
_output_shapes
:@2
readin_4/Tanh?
$Policy_layer_1/Conv2D/ReadVariableOpReadVariableOp-policy_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Policy_layer_1/Conv2D/ReadVariableOp?
Policy_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Policy_layer_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Policy_layer_1/Conv2D?
%Policy_layer_1/BiasAdd/ReadVariableOpReadVariableOp.policy_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_layer_1/BiasAdd/ReadVariableOp?
Policy_layer_1/BiasAddBiasAddPolicy_layer_1/Conv2D:output:0-Policy_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
Policy_layer_1/BiasAdd?
Policy_layer_1/TanhTanhPolicy_layer_1/BiasAdd:output:0*
T0*&
_output_shapes
:2
Policy_layer_1/Tanhw
flatten_726/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_726/Const?
flatten_726/ReshapeReshapePolicy_layer_1/Tanh:y:0flatten_726/Const:output:0*
T0*
_output_shapes
:	?2
flatten_726/Reshape?
$Policy_readout/MatMul/ReadVariableOpReadVariableOp-policy_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Policy_readout/MatMul/ReadVariableOp?
Policy_readout/MatMulMatMulflatten_726/Reshape:output:0,Policy_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_readout/MatMul?
%Policy_readout/BiasAdd/ReadVariableOpReadVariableOp.policy_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_readout/BiasAdd/ReadVariableOp?
Policy_readout/BiasAddBiasAddPolicy_readout/MatMul:product:0-Policy_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_readout/BiasAdd?
Policy_readout/SoftmaxSoftmaxPolicy_readout/BiasAdd:output:0*
T0*
_output_shapes

:2
Policy_readout/Softmaxd
SqueezeSqueeze Policy_readout/Softmax:softmax:0*
T0*
_output_shapes
:2	
Squeeze?
$Critic_layer_1/Conv2D/ReadVariableOpReadVariableOp-critic_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Critic_layer_1/Conv2D/ReadVariableOp?
Critic_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Critic_layer_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
2
Critic_layer_1/Conv2D?
%Critic_layer_1/BiasAdd/ReadVariableOpReadVariableOp.critic_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_layer_1/BiasAdd/ReadVariableOp?
Critic_layer_1/BiasAddBiasAddCritic_layer_1/Conv2D:output:0-Critic_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
Critic_layer_1/BiasAdd?
Critic_layer_1/TanhTanhCritic_layer_1/BiasAdd:output:0*
T0*&
_output_shapes
:2
Critic_layer_1/Tanhw
flatten_727/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_727/Const?
flatten_727/ReshapeReshapeCritic_layer_1/Tanh:y:0flatten_727/Const:output:0*
T0*
_output_shapes
:	?2
flatten_727/Reshape?
$Critic_readout/MatMul/ReadVariableOpReadVariableOp-critic_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Critic_readout/MatMul/ReadVariableOp?
Critic_readout/MatMulMatMulflatten_727/Reshape:output:0,Critic_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic_readout/MatMul?
%Critic_readout/BiasAdd/ReadVariableOpReadVariableOp.critic_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_readout/BiasAdd/ReadVariableOp?
Critic_readout/BiasAddBiasAddCritic_readout/MatMul:product:0-Critic_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Critic_readout/BiasAdd|
Critic_readout/TanhTanhCritic_readout/BiasAdd:output:0*
T0*
_output_shapes

:2
Critic_readout/Tanh[
	Squeeze_1SqueezeCritic_readout/Tanh:y:0*
T0*
_output_shapes
: 2
	Squeeze_1?
IdentityIdentitySqueeze:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
:2

Identity?

Identity_1IdentitySqueeze_1:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*e
_input_shapesT
R:::::::::::::::::2N
%Critic_layer_1/BiasAdd/ReadVariableOp%Critic_layer_1/BiasAdd/ReadVariableOp2L
$Critic_layer_1/Conv2D/ReadVariableOp$Critic_layer_1/Conv2D/ReadVariableOp2N
%Critic_readout/BiasAdd/ReadVariableOp%Critic_readout/BiasAdd/ReadVariableOp2L
$Critic_readout/MatMul/ReadVariableOp$Critic_readout/MatMul/ReadVariableOp2N
%Policy_layer_1/BiasAdd/ReadVariableOp%Policy_layer_1/BiasAdd/ReadVariableOp2L
$Policy_layer_1/Conv2D/ReadVariableOp$Policy_layer_1/Conv2D/ReadVariableOp2N
%Policy_readout/BiasAdd/ReadVariableOp%Policy_readout/BiasAdd/ReadVariableOp2L
$Policy_readout/MatMul/ReadVariableOp$Policy_readout/MatMul/ReadVariableOp2B
readin_1/BiasAdd/ReadVariableOpreadin_1/BiasAdd/ReadVariableOp2@
readin_1/Conv2D/ReadVariableOpreadin_1/Conv2D/ReadVariableOp2B
readin_2/BiasAdd/ReadVariableOpreadin_2/BiasAdd/ReadVariableOp2@
readin_2/Conv2D/ReadVariableOpreadin_2/Conv2D/ReadVariableOp2B
readin_3/BiasAdd/ReadVariableOpreadin_3/BiasAdd/ReadVariableOp2@
readin_3/Conv2D/ReadVariableOpreadin_3/Conv2D/ReadVariableOp2B
readin_4/BiasAdd/ReadVariableOpreadin_4/BiasAdd/ReadVariableOp2@
readin_4/Conv2D/ReadVariableOpreadin_4/Conv2D/ReadVariableOp:S O
&
_output_shapes
:
%
_user_specified_nameobservation
?
?
1__inference_Critic_layer_1_layer_call_fn_96166510

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_961660642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_readin_1_layer_call_and_return_conditional_losses_96166390

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?^
?
__inference_call_96165835
observation+
'readin_1_conv2d_readvariableop_resource,
(readin_1_biasadd_readvariableop_resource+
'readin_2_conv2d_readvariableop_resource,
(readin_2_biasadd_readvariableop_resource+
'readin_3_conv2d_readvariableop_resource,
(readin_3_biasadd_readvariableop_resource+
'readin_4_conv2d_readvariableop_resource,
(readin_4_biasadd_readvariableop_resource1
-policy_layer_1_conv2d_readvariableop_resource2
.policy_layer_1_biasadd_readvariableop_resource1
-policy_readout_matmul_readvariableop_resource2
.policy_readout_biasadd_readvariableop_resource1
-critic_layer_1_conv2d_readvariableop_resource2
.critic_layer_1_biasadd_readvariableop_resource1
-critic_readout_matmul_readvariableop_resource2
.critic_readout_biasadd_readvariableop_resource
identity

identity_1??%Critic_layer_1/BiasAdd/ReadVariableOp?$Critic_layer_1/Conv2D/ReadVariableOp?%Critic_readout/BiasAdd/ReadVariableOp?$Critic_readout/MatMul/ReadVariableOp?%Policy_layer_1/BiasAdd/ReadVariableOp?$Policy_layer_1/Conv2D/ReadVariableOp?%Policy_readout/BiasAdd/ReadVariableOp?$Policy_readout/MatMul/ReadVariableOp?readin_1/BiasAdd/ReadVariableOp?readin_1/Conv2D/ReadVariableOp?readin_2/BiasAdd/ReadVariableOp?readin_2/Conv2D/ReadVariableOp?readin_3/BiasAdd/ReadVariableOp?readin_3/Conv2D/ReadVariableOp?readin_4/BiasAdd/ReadVariableOp?readin_4/Conv2D/ReadVariableOp?
readin_1/Conv2D/ReadVariableOpReadVariableOp'readin_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
readin_1/Conv2D/ReadVariableOp?
readin_1/Conv2DConv2Dobservation&readin_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_1/Conv2D?
readin_1/BiasAdd/ReadVariableOpReadVariableOp(readin_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_1/BiasAdd/ReadVariableOp?
readin_1/BiasAddBiasAddreadin_1/Conv2D:output:0'readin_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_1/BiasAdd{
readin_1/TanhTanhreadin_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_1/Tanh?
readin_2/Conv2D/ReadVariableOpReadVariableOp'readin_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_2/Conv2D/ReadVariableOp?
readin_2/Conv2DConv2Dreadin_1/Tanh:y:0&readin_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_2/Conv2D?
readin_2/BiasAdd/ReadVariableOpReadVariableOp(readin_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_2/BiasAdd/ReadVariableOp?
readin_2/BiasAddBiasAddreadin_2/Conv2D:output:0'readin_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_2/BiasAdd{
readin_2/TanhTanhreadin_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_2/Tanh?
readin_3/Conv2D/ReadVariableOpReadVariableOp'readin_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_3/Conv2D/ReadVariableOp?
readin_3/Conv2DConv2Dreadin_2/Tanh:y:0&readin_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_3/Conv2D?
readin_3/BiasAdd/ReadVariableOpReadVariableOp(readin_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_3/BiasAdd/ReadVariableOp?
readin_3/BiasAddBiasAddreadin_3/Conv2D:output:0'readin_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_3/BiasAdd{
readin_3/TanhTanhreadin_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_3/Tanh?
readin_4/Conv2D/ReadVariableOpReadVariableOp'readin_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_4/Conv2D/ReadVariableOp?
readin_4/Conv2DConv2Dreadin_3/Tanh:y:0&readin_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_4/Conv2D?
readin_4/BiasAdd/ReadVariableOpReadVariableOp(readin_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_4/BiasAdd/ReadVariableOp?
readin_4/BiasAddBiasAddreadin_4/Conv2D:output:0'readin_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_4/BiasAdd{
readin_4/TanhTanhreadin_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_4/Tanh?
$Policy_layer_1/Conv2D/ReadVariableOpReadVariableOp-policy_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Policy_layer_1/Conv2D/ReadVariableOp?
Policy_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Policy_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Policy_layer_1/Conv2D?
%Policy_layer_1/BiasAdd/ReadVariableOpReadVariableOp.policy_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_layer_1/BiasAdd/ReadVariableOp?
Policy_layer_1/BiasAddBiasAddPolicy_layer_1/Conv2D:output:0-Policy_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Policy_layer_1/BiasAdd?
Policy_layer_1/TanhTanhPolicy_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Policy_layer_1/Tanhw
flatten_726/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_726/Const?
flatten_726/ReshapeReshapePolicy_layer_1/Tanh:y:0flatten_726/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_726/Reshape?
$Policy_readout/MatMul/ReadVariableOpReadVariableOp-policy_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Policy_readout/MatMul/ReadVariableOp?
Policy_readout/MatMulMatMulflatten_726/Reshape:output:0,Policy_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Policy_readout/MatMul?
%Policy_readout/BiasAdd/ReadVariableOpReadVariableOp.policy_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_readout/BiasAdd/ReadVariableOp?
Policy_readout/BiasAddBiasAddPolicy_readout/MatMul:product:0-Policy_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Policy_readout/BiasAdd?
Policy_readout/SoftmaxSoftmaxPolicy_readout/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Policy_readout/Softmaxb
SqueezeSqueeze Policy_readout/Softmax:softmax:0*
T0*
_output_shapes
:2	
Squeeze?
$Critic_layer_1/Conv2D/ReadVariableOpReadVariableOp-critic_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Critic_layer_1/Conv2D/ReadVariableOp?
Critic_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Critic_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Critic_layer_1/Conv2D?
%Critic_layer_1/BiasAdd/ReadVariableOpReadVariableOp.critic_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_layer_1/BiasAdd/ReadVariableOp?
Critic_layer_1/BiasAddBiasAddCritic_layer_1/Conv2D:output:0-Critic_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Critic_layer_1/BiasAdd?
Critic_layer_1/TanhTanhCritic_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Critic_layer_1/Tanhw
flatten_727/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_727/Const?
flatten_727/ReshapeReshapeCritic_layer_1/Tanh:y:0flatten_727/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_727/Reshape?
$Critic_readout/MatMul/ReadVariableOpReadVariableOp-critic_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Critic_readout/MatMul/ReadVariableOp?
Critic_readout/MatMulMatMulflatten_727/Reshape:output:0,Critic_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic_readout/MatMul?
%Critic_readout/BiasAdd/ReadVariableOpReadVariableOp.critic_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_readout/BiasAdd/ReadVariableOp?
Critic_readout/BiasAddBiasAddCritic_readout/MatMul:product:0-Critic_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic_readout/BiasAdd?
Critic_readout/TanhTanhCritic_readout/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Critic_readout/Tanh]
	Squeeze_1SqueezeCritic_readout/Tanh:y:0*
T0*
_output_shapes
:2
	Squeeze_1?
IdentityIdentitySqueeze:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
:2

Identity?

Identity_1IdentitySqueeze_1:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2N
%Critic_layer_1/BiasAdd/ReadVariableOp%Critic_layer_1/BiasAdd/ReadVariableOp2L
$Critic_layer_1/Conv2D/ReadVariableOp$Critic_layer_1/Conv2D/ReadVariableOp2N
%Critic_readout/BiasAdd/ReadVariableOp%Critic_readout/BiasAdd/ReadVariableOp2L
$Critic_readout/MatMul/ReadVariableOp$Critic_readout/MatMul/ReadVariableOp2N
%Policy_layer_1/BiasAdd/ReadVariableOp%Policy_layer_1/BiasAdd/ReadVariableOp2L
$Policy_layer_1/Conv2D/ReadVariableOp$Policy_layer_1/Conv2D/ReadVariableOp2N
%Policy_readout/BiasAdd/ReadVariableOp%Policy_readout/BiasAdd/ReadVariableOp2L
$Policy_readout/MatMul/ReadVariableOp$Policy_readout/MatMul/ReadVariableOp2B
readin_1/BiasAdd/ReadVariableOpreadin_1/BiasAdd/ReadVariableOp2@
readin_1/Conv2D/ReadVariableOpreadin_1/Conv2D/ReadVariableOp2B
readin_2/BiasAdd/ReadVariableOpreadin_2/BiasAdd/ReadVariableOp2@
readin_2/Conv2D/ReadVariableOpreadin_2/Conv2D/ReadVariableOp2B
readin_3/BiasAdd/ReadVariableOpreadin_3/BiasAdd/ReadVariableOp2@
readin_3/Conv2D/ReadVariableOpreadin_3/Conv2D/ReadVariableOp2B
readin_4/BiasAdd/ReadVariableOpreadin_4/BiasAdd/ReadVariableOp2@
readin_4/Conv2D/ReadVariableOpreadin_4/Conv2D/ReadVariableOp:\ X
/
_output_shapes
:?????????
%
_user_specified_nameobservation
?

?
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_96166064

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_readin_3_layer_call_and_return_conditional_losses_96166430

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
1__inference_Policy_readout_layer_call_fn_96166359

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Policy_readout_layer_call_and_return_conditional_losses_961660362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
1__inference_Policy_layer_1_layer_call_fn_96166479

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_961659952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?8
?
N__inference_actor_critic_363_layer_call_and_return_conditional_losses_96166124
input_1
readin_1_96165898
readin_1_96165900
readin_2_96165925
readin_2_96165927
readin_3_96165952
readin_3_96165954
readin_4_96165979
readin_4_96165981
policy_layer_1_96166006
policy_layer_1_96166008
policy_readout_96166047
policy_readout_96166049
critic_layer_1_96166075
critic_layer_1_96166077
critic_readout_96166116
critic_readout_96166118
identity

identity_1??&Critic_layer_1/StatefulPartitionedCall?&Critic_readout/StatefulPartitionedCall?&Policy_layer_1/StatefulPartitionedCall?&Policy_readout/StatefulPartitionedCall? readin_1/StatefulPartitionedCall? readin_2/StatefulPartitionedCall? readin_3/StatefulPartitionedCall? readin_4/StatefulPartitionedCall?
 readin_1/StatefulPartitionedCallStatefulPartitionedCallinput_1readin_1_96165898readin_1_96165900*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_1_layer_call_and_return_conditional_losses_961658872"
 readin_1/StatefulPartitionedCall?
 readin_2/StatefulPartitionedCallStatefulPartitionedCall)readin_1/StatefulPartitionedCall:output:0readin_2_96165925readin_2_96165927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_2_layer_call_and_return_conditional_losses_961659142"
 readin_2/StatefulPartitionedCall?
 readin_3/StatefulPartitionedCallStatefulPartitionedCall)readin_2/StatefulPartitionedCall:output:0readin_3_96165952readin_3_96165954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_3_layer_call_and_return_conditional_losses_961659412"
 readin_3/StatefulPartitionedCall?
 readin_4/StatefulPartitionedCallStatefulPartitionedCall)readin_3/StatefulPartitionedCall:output:0readin_4_96165979readin_4_96165981*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_4_layer_call_and_return_conditional_losses_961659682"
 readin_4/StatefulPartitionedCall?
&Policy_layer_1/StatefulPartitionedCallStatefulPartitionedCall)readin_4/StatefulPartitionedCall:output:0policy_layer_1_96166006policy_layer_1_96166008*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_961659952(
&Policy_layer_1/StatefulPartitionedCall?
flatten_726/PartitionedCallPartitionedCall/Policy_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flatten_726_layer_call_and_return_conditional_losses_961660172
flatten_726/PartitionedCall?
&Policy_readout/StatefulPartitionedCallStatefulPartitionedCall$flatten_726/PartitionedCall:output:0policy_readout_96166047policy_readout_96166049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Policy_readout_layer_call_and_return_conditional_losses_961660362(
&Policy_readout/StatefulPartitionedCallq
SqueezeSqueeze/Policy_readout/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2	
Squeeze?
&Critic_layer_1/StatefulPartitionedCallStatefulPartitionedCall)readin_4/StatefulPartitionedCall:output:0critic_layer_1_96166075critic_layer_1_96166077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_961660642(
&Critic_layer_1/StatefulPartitionedCall?
flatten_727/PartitionedCallPartitionedCall/Critic_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flatten_727_layer_call_and_return_conditional_losses_961660862
flatten_727/PartitionedCall?
&Critic_readout/StatefulPartitionedCallStatefulPartitionedCall$flatten_727/PartitionedCall:output:0critic_readout_96166116critic_readout_96166118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Critic_readout_layer_call_and_return_conditional_losses_961661052(
&Critic_readout/StatefulPartitionedCallu
	Squeeze_1Squeeze/Critic_readout/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
	Squeeze_1?
IdentityIdentitySqueeze:output:0'^Critic_layer_1/StatefulPartitionedCall'^Critic_readout/StatefulPartitionedCall'^Policy_layer_1/StatefulPartitionedCall'^Policy_readout/StatefulPartitionedCall!^readin_1/StatefulPartitionedCall!^readin_2/StatefulPartitionedCall!^readin_3/StatefulPartitionedCall!^readin_4/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity?

Identity_1IdentitySqueeze_1:output:0'^Critic_layer_1/StatefulPartitionedCall'^Critic_readout/StatefulPartitionedCall'^Policy_layer_1/StatefulPartitionedCall'^Policy_readout/StatefulPartitionedCall!^readin_1/StatefulPartitionedCall!^readin_2/StatefulPartitionedCall!^readin_3/StatefulPartitionedCall!^readin_4/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2P
&Critic_layer_1/StatefulPartitionedCall&Critic_layer_1/StatefulPartitionedCall2P
&Critic_readout/StatefulPartitionedCall&Critic_readout/StatefulPartitionedCall2P
&Policy_layer_1/StatefulPartitionedCall&Policy_layer_1/StatefulPartitionedCall2P
&Policy_readout/StatefulPartitionedCall&Policy_readout/StatefulPartitionedCall2D
 readin_1/StatefulPartitionedCall readin_1/StatefulPartitionedCall2D
 readin_2/StatefulPartitionedCall readin_2/StatefulPartitionedCall2D
 readin_3/StatefulPartitionedCall readin_3/StatefulPartitionedCall2D
 readin_4/StatefulPartitionedCall readin_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
+__inference_readin_1_layer_call_fn_96166399

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_1_layer_call_and_return_conditional_losses_961658872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_96166501

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_readin_4_layer_call_and_return_conditional_losses_96165968

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
F__inference_readin_1_layer_call_and_return_conditional_losses_96165887

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_readin_2_layer_call_and_return_conditional_losses_96165914

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
e
I__inference_flatten_727_layer_call_and_return_conditional_losses_96166516

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_readin_4_layer_call_fn_96166459

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_4_layer_call_and_return_conditional_losses_961659682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?^
?
__inference_call_96166339
observation+
'readin_1_conv2d_readvariableop_resource,
(readin_1_biasadd_readvariableop_resource+
'readin_2_conv2d_readvariableop_resource,
(readin_2_biasadd_readvariableop_resource+
'readin_3_conv2d_readvariableop_resource,
(readin_3_biasadd_readvariableop_resource+
'readin_4_conv2d_readvariableop_resource,
(readin_4_biasadd_readvariableop_resource1
-policy_layer_1_conv2d_readvariableop_resource2
.policy_layer_1_biasadd_readvariableop_resource1
-policy_readout_matmul_readvariableop_resource2
.policy_readout_biasadd_readvariableop_resource1
-critic_layer_1_conv2d_readvariableop_resource2
.critic_layer_1_biasadd_readvariableop_resource1
-critic_readout_matmul_readvariableop_resource2
.critic_readout_biasadd_readvariableop_resource
identity

identity_1??%Critic_layer_1/BiasAdd/ReadVariableOp?$Critic_layer_1/Conv2D/ReadVariableOp?%Critic_readout/BiasAdd/ReadVariableOp?$Critic_readout/MatMul/ReadVariableOp?%Policy_layer_1/BiasAdd/ReadVariableOp?$Policy_layer_1/Conv2D/ReadVariableOp?%Policy_readout/BiasAdd/ReadVariableOp?$Policy_readout/MatMul/ReadVariableOp?readin_1/BiasAdd/ReadVariableOp?readin_1/Conv2D/ReadVariableOp?readin_2/BiasAdd/ReadVariableOp?readin_2/Conv2D/ReadVariableOp?readin_3/BiasAdd/ReadVariableOp?readin_3/Conv2D/ReadVariableOp?readin_4/BiasAdd/ReadVariableOp?readin_4/Conv2D/ReadVariableOp?
readin_1/Conv2D/ReadVariableOpReadVariableOp'readin_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
readin_1/Conv2D/ReadVariableOp?
readin_1/Conv2DConv2Dobservation&readin_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_1/Conv2D?
readin_1/BiasAdd/ReadVariableOpReadVariableOp(readin_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_1/BiasAdd/ReadVariableOp?
readin_1/BiasAddBiasAddreadin_1/Conv2D:output:0'readin_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_1/BiasAdd{
readin_1/TanhTanhreadin_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_1/Tanh?
readin_2/Conv2D/ReadVariableOpReadVariableOp'readin_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_2/Conv2D/ReadVariableOp?
readin_2/Conv2DConv2Dreadin_1/Tanh:y:0&readin_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_2/Conv2D?
readin_2/BiasAdd/ReadVariableOpReadVariableOp(readin_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_2/BiasAdd/ReadVariableOp?
readin_2/BiasAddBiasAddreadin_2/Conv2D:output:0'readin_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_2/BiasAdd{
readin_2/TanhTanhreadin_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_2/Tanh?
readin_3/Conv2D/ReadVariableOpReadVariableOp'readin_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_3/Conv2D/ReadVariableOp?
readin_3/Conv2DConv2Dreadin_2/Tanh:y:0&readin_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_3/Conv2D?
readin_3/BiasAdd/ReadVariableOpReadVariableOp(readin_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_3/BiasAdd/ReadVariableOp?
readin_3/BiasAddBiasAddreadin_3/Conv2D:output:0'readin_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_3/BiasAdd{
readin_3/TanhTanhreadin_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_3/Tanh?
readin_4/Conv2D/ReadVariableOpReadVariableOp'readin_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
readin_4/Conv2D/ReadVariableOp?
readin_4/Conv2DConv2Dreadin_3/Tanh:y:0&readin_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
readin_4/Conv2D?
readin_4/BiasAdd/ReadVariableOpReadVariableOp(readin_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
readin_4/BiasAdd/ReadVariableOp?
readin_4/BiasAddBiasAddreadin_4/Conv2D:output:0'readin_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
readin_4/BiasAdd{
readin_4/TanhTanhreadin_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
readin_4/Tanh?
$Policy_layer_1/Conv2D/ReadVariableOpReadVariableOp-policy_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Policy_layer_1/Conv2D/ReadVariableOp?
Policy_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Policy_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Policy_layer_1/Conv2D?
%Policy_layer_1/BiasAdd/ReadVariableOpReadVariableOp.policy_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_layer_1/BiasAdd/ReadVariableOp?
Policy_layer_1/BiasAddBiasAddPolicy_layer_1/Conv2D:output:0-Policy_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Policy_layer_1/BiasAdd?
Policy_layer_1/TanhTanhPolicy_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Policy_layer_1/Tanhw
flatten_726/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_726/Const?
flatten_726/ReshapeReshapePolicy_layer_1/Tanh:y:0flatten_726/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_726/Reshape?
$Policy_readout/MatMul/ReadVariableOpReadVariableOp-policy_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Policy_readout/MatMul/ReadVariableOp?
Policy_readout/MatMulMatMulflatten_726/Reshape:output:0,Policy_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Policy_readout/MatMul?
%Policy_readout/BiasAdd/ReadVariableOpReadVariableOp.policy_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Policy_readout/BiasAdd/ReadVariableOp?
Policy_readout/BiasAddBiasAddPolicy_readout/MatMul:product:0-Policy_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Policy_readout/BiasAdd?
Policy_readout/SoftmaxSoftmaxPolicy_readout/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Policy_readout/Softmaxb
SqueezeSqueeze Policy_readout/Softmax:softmax:0*
T0*
_output_shapes
:2	
Squeeze?
$Critic_layer_1/Conv2D/ReadVariableOpReadVariableOp-critic_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$Critic_layer_1/Conv2D/ReadVariableOp?
Critic_layer_1/Conv2DConv2Dreadin_4/Tanh:y:0,Critic_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Critic_layer_1/Conv2D?
%Critic_layer_1/BiasAdd/ReadVariableOpReadVariableOp.critic_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_layer_1/BiasAdd/ReadVariableOp?
Critic_layer_1/BiasAddBiasAddCritic_layer_1/Conv2D:output:0-Critic_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Critic_layer_1/BiasAdd?
Critic_layer_1/TanhTanhCritic_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Critic_layer_1/Tanhw
flatten_727/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_727/Const?
flatten_727/ReshapeReshapeCritic_layer_1/Tanh:y:0flatten_727/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_727/Reshape?
$Critic_readout/MatMul/ReadVariableOpReadVariableOp-critic_readout_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$Critic_readout/MatMul/ReadVariableOp?
Critic_readout/MatMulMatMulflatten_727/Reshape:output:0,Critic_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic_readout/MatMul?
%Critic_readout/BiasAdd/ReadVariableOpReadVariableOp.critic_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Critic_readout/BiasAdd/ReadVariableOp?
Critic_readout/BiasAddBiasAddCritic_readout/MatMul:product:0-Critic_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Critic_readout/BiasAdd?
Critic_readout/TanhTanhCritic_readout/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Critic_readout/Tanh]
	Squeeze_1SqueezeCritic_readout/Tanh:y:0*
T0*
_output_shapes
:2
	Squeeze_1?
IdentityIdentitySqueeze:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
:2

Identity?

Identity_1IdentitySqueeze_1:output:0&^Critic_layer_1/BiasAdd/ReadVariableOp%^Critic_layer_1/Conv2D/ReadVariableOp&^Critic_readout/BiasAdd/ReadVariableOp%^Critic_readout/MatMul/ReadVariableOp&^Policy_layer_1/BiasAdd/ReadVariableOp%^Policy_layer_1/Conv2D/ReadVariableOp&^Policy_readout/BiasAdd/ReadVariableOp%^Policy_readout/MatMul/ReadVariableOp ^readin_1/BiasAdd/ReadVariableOp^readin_1/Conv2D/ReadVariableOp ^readin_2/BiasAdd/ReadVariableOp^readin_2/Conv2D/ReadVariableOp ^readin_3/BiasAdd/ReadVariableOp^readin_3/Conv2D/ReadVariableOp ^readin_4/BiasAdd/ReadVariableOp^readin_4/Conv2D/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2N
%Critic_layer_1/BiasAdd/ReadVariableOp%Critic_layer_1/BiasAdd/ReadVariableOp2L
$Critic_layer_1/Conv2D/ReadVariableOp$Critic_layer_1/Conv2D/ReadVariableOp2N
%Critic_readout/BiasAdd/ReadVariableOp%Critic_readout/BiasAdd/ReadVariableOp2L
$Critic_readout/MatMul/ReadVariableOp$Critic_readout/MatMul/ReadVariableOp2N
%Policy_layer_1/BiasAdd/ReadVariableOp%Policy_layer_1/BiasAdd/ReadVariableOp2L
$Policy_layer_1/Conv2D/ReadVariableOp$Policy_layer_1/Conv2D/ReadVariableOp2N
%Policy_readout/BiasAdd/ReadVariableOp%Policy_readout/BiasAdd/ReadVariableOp2L
$Policy_readout/MatMul/ReadVariableOp$Policy_readout/MatMul/ReadVariableOp2B
readin_1/BiasAdd/ReadVariableOpreadin_1/BiasAdd/ReadVariableOp2@
readin_1/Conv2D/ReadVariableOpreadin_1/Conv2D/ReadVariableOp2B
readin_2/BiasAdd/ReadVariableOpreadin_2/BiasAdd/ReadVariableOp2@
readin_2/Conv2D/ReadVariableOpreadin_2/Conv2D/ReadVariableOp2B
readin_3/BiasAdd/ReadVariableOpreadin_3/BiasAdd/ReadVariableOp2@
readin_3/Conv2D/ReadVariableOpreadin_3/Conv2D/ReadVariableOp2B
readin_4/BiasAdd/ReadVariableOpreadin_4/BiasAdd/ReadVariableOp2@
readin_4/Conv2D/ReadVariableOpreadin_4/Conv2D/ReadVariableOp:\ X
/
_output_shapes
:?????????
%
_user_specified_nameobservation
?*
?
!__inference__traced_save_96166593
file_prefix4
0savev2_policy_readout_kernel_read_readvariableop2
.savev2_policy_readout_bias_read_readvariableop4
0savev2_critic_readout_kernel_read_readvariableop2
.savev2_critic_readout_bias_read_readvariableop.
*savev2_readin_1_kernel_read_readvariableop,
(savev2_readin_1_bias_read_readvariableop.
*savev2_readin_2_kernel_read_readvariableop,
(savev2_readin_2_bias_read_readvariableop.
*savev2_readin_3_kernel_read_readvariableop,
(savev2_readin_3_bias_read_readvariableop.
*savev2_readin_4_kernel_read_readvariableop,
(savev2_readin_4_bias_read_readvariableop4
0savev2_policy_layer_1_kernel_read_readvariableop2
.savev2_policy_layer_1_bias_read_readvariableop4
0savev2_critic_layer_1_kernel_read_readvariableop2
.savev2_critic_layer_1_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0readout_policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB.readout_policy/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_policy_readout_kernel_read_readvariableop.savev2_policy_readout_bias_read_readvariableop0savev2_critic_readout_kernel_read_readvariableop.savev2_critic_readout_bias_read_readvariableop*savev2_readin_1_kernel_read_readvariableop(savev2_readin_1_bias_read_readvariableop*savev2_readin_2_kernel_read_readvariableop(savev2_readin_2_bias_read_readvariableop*savev2_readin_3_kernel_read_readvariableop(savev2_readin_3_bias_read_readvariableop*savev2_readin_4_kernel_read_readvariableop(savev2_readin_4_bias_read_readvariableop0savev2_policy_layer_1_kernel_read_readvariableop.savev2_policy_layer_1_bias_read_readvariableop0savev2_critic_layer_1_kernel_read_readvariableop.savev2_critic_layer_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?::	?::@:@:@@:@:@@:@:@@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
::

_output_shapes
: 
?
?
1__inference_Critic_readout_layer_call_fn_96166379

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_Critic_readout_layer_call_and_return_conditional_losses_961661052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_actor_critic_363_layer_call_fn_96166164
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_actor_critic_363_layer_call_and_return_conditional_losses_961661242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
L__inference_Policy_readout_layer_call_and_return_conditional_losses_96166350

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_readin_2_layer_call_and_return_conditional_losses_96166410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_96166205
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_961658722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_96165995

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
L__inference_Policy_readout_layer_call_and_return_conditional_losses_96166036

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_readin_3_layer_call_fn_96166439

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_readin_3_layer_call_and_return_conditional_losses_961659412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?D
?
$__inference__traced_restore_96166651
file_prefix*
&assignvariableop_policy_readout_kernel*
&assignvariableop_1_policy_readout_bias,
(assignvariableop_2_critic_readout_kernel*
&assignvariableop_3_critic_readout_bias&
"assignvariableop_4_readin_1_kernel$
 assignvariableop_5_readin_1_bias&
"assignvariableop_6_readin_2_kernel$
 assignvariableop_7_readin_2_bias&
"assignvariableop_8_readin_3_kernel$
 assignvariableop_9_readin_3_bias'
#assignvariableop_10_readin_4_kernel%
!assignvariableop_11_readin_4_bias-
)assignvariableop_12_policy_layer_1_kernel+
'assignvariableop_13_policy_layer_1_bias-
)assignvariableop_14_critic_layer_1_kernel+
'assignvariableop_15_critic_layer_1_bias
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0readout_policy/kernel/.ATTRIBUTES/VARIABLE_VALUEB.readout_policy/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp&assignvariableop_policy_readout_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_policy_readout_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_critic_readout_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_critic_readout_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_readin_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_readin_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_readin_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_readin_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_readin_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_readin_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_readin_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_readin_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_policy_layer_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_policy_layer_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_critic_layer_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_critic_layer_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
L__inference_Critic_readout_layer_call_and_return_conditional_losses_96166370

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_readin_3_layer_call_and_return_conditional_losses_96165941

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
TanhTanhBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
#__inference__wrapped_model_96165872
input_1
actor_critic_363_96165836
actor_critic_363_96165838
actor_critic_363_96165840
actor_critic_363_96165842
actor_critic_363_96165844
actor_critic_363_96165846
actor_critic_363_96165848
actor_critic_363_96165850
actor_critic_363_96165852
actor_critic_363_96165854
actor_critic_363_96165856
actor_critic_363_96165858
actor_critic_363_96165860
actor_critic_363_96165862
actor_critic_363_96165864
actor_critic_363_96165866
identity

identity_1??(actor_critic_363/StatefulPartitionedCall?
(actor_critic_363/StatefulPartitionedCallStatefulPartitionedCallinput_1actor_critic_363_96165836actor_critic_363_96165838actor_critic_363_96165840actor_critic_363_96165842actor_critic_363_96165844actor_critic_363_96165846actor_critic_363_96165848actor_critic_363_96165850actor_critic_363_96165852actor_critic_363_96165854actor_critic_363_96165856actor_critic_363_96165858actor_critic_363_96165860actor_critic_363_96165862actor_critic_363_96165864actor_critic_363_96165866*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_call_961658352*
(actor_critic_363/StatefulPartitionedCall?
IdentityIdentity1actor_critic_363/StatefulPartitionedCall:output:0)^actor_critic_363/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity?

Identity_1Identity1actor_critic_363/StatefulPartitionedCall:output:1)^actor_critic_363/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2T
(actor_critic_363/StatefulPartitionedCall(actor_critic_363/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????+
policy!
StatefulPartitionedCall:03
value_estimate!
StatefulPartitionedCall:1tensorflow/serving/predict:??
?
readin_layers
policy_layers
readout_policy
value_layers
readout_value
	variables
regularization_losses
trainable_variables
		keras_api


signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__
	?call"?
_tf_keras_model?{"class_name": "ActorCritic", "name": "actor_critic_363", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "ActorCritic"}}
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Policy_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_readout", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 726}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 726]}}
.
0
1"
trackable_list_wrapper
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "Critic_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic_readout", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 726}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 726]}}
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
10
11
)12
*13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
10
11
)12
*13
14
15"
trackable_list_wrapper
?
	variables
+layer_metrics
regularization_losses
,layer_regularization_losses
trainable_variables
-metrics

.layers
/non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

kernel
 bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "readin_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "readin_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 21}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 21]}}
?	

!kernel
"bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "readin_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "readin_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?	

#kernel
$bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "readin_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "readin_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?	

%kernel
&bias
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "readin_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "readin_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?	

'kernel
(bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Policy_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_layer_1", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_726", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_726", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
(:&	?2Policy_readout/kernel
!:2Policy_readout/bias
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
?
	variables
Hlayer_metrics
regularization_losses
Ilayer_regularization_losses
trainable_variables
Jmetrics

Klayers
Lnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

)kernel
*bias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "Critic_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Critic_layer_1", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}
?
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_727", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_727", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
(:&	?2Critic_readout/kernel
!:2Critic_readout/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
Ulayer_metrics
regularization_losses
Vlayer_regularization_losses
trainable_variables
Wmetrics

Xlayers
Ynon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@2readin_1/kernel
:@2readin_1/bias
):'@@2readin_2/kernel
:@2readin_2/bias
):'@@2readin_3/kernel
:@2readin_3/bias
):'@@2readin_4/kernel
:@2readin_4/bias
/:-@2Policy_layer_1/kernel
!:2Policy_layer_1/bias
/:-@2Critic_layer_1/kernel
!:2Critic_layer_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
0	variables
Zlayer_metrics
1regularization_losses
[layer_regularization_losses
2trainable_variables
\metrics

]layers
^non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
4	variables
_layer_metrics
5regularization_losses
`layer_regularization_losses
6trainable_variables
ametrics

blayers
cnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
8	variables
dlayer_metrics
9regularization_losses
elayer_regularization_losses
:trainable_variables
fmetrics

glayers
hnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
<	variables
ilayer_metrics
=regularization_losses
jlayer_regularization_losses
>trainable_variables
kmetrics

llayers
mnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
@	variables
nlayer_metrics
Aregularization_losses
olayer_regularization_losses
Btrainable_variables
pmetrics

qlayers
rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
D	variables
slayer_metrics
Eregularization_losses
tlayer_regularization_losses
Ftrainable_variables
umetrics

vlayers
wnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
M	variables
xlayer_metrics
Nregularization_losses
ylayer_regularization_losses
Otrainable_variables
zmetrics

{layers
|non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Q	variables
}layer_metrics
Rregularization_losses
~layer_regularization_losses
Strainable_variables
metrics
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?2?
#__inference__wrapped_model_96165872?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
N__inference_actor_critic_363_layer_call_and_return_conditional_losses_96166124?
???
FullArgSpec"
args?
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
3__inference_actor_critic_363_layer_call_fn_96166164?
???
FullArgSpec"
args?
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
__inference_call_96166272
__inference_call_96166339?
???
FullArgSpec"
args?
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_Policy_readout_layer_call_and_return_conditional_losses_96166350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_Policy_readout_layer_call_fn_96166359?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_Critic_readout_layer_call_and_return_conditional_losses_96166370?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_Critic_readout_layer_call_fn_96166379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_96166205input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_readin_1_layer_call_and_return_conditional_losses_96166390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_readin_1_layer_call_fn_96166399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_readin_2_layer_call_and_return_conditional_losses_96166410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_readin_2_layer_call_fn_96166419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_readin_3_layer_call_and_return_conditional_losses_96166430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_readin_3_layer_call_fn_96166439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_readin_4_layer_call_and_return_conditional_losses_96166450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_readin_4_layer_call_fn_96166459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_96166470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_Policy_layer_1_layer_call_fn_96166479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_726_layer_call_and_return_conditional_losses_96166485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_726_layer_call_fn_96166490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_96166501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_Critic_layer_1_layer_call_fn_96166510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_727_layer_call_and_return_conditional_losses_96166516?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_flatten_727_layer_call_fn_96166521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
L__inference_Critic_layer_1_layer_call_and_return_conditional_losses_96166501l)*7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????
? ?
1__inference_Critic_layer_1_layer_call_fn_96166510_)*7?4
-?*
(?%
inputs?????????@
? " ???????????
L__inference_Critic_readout_layer_call_and_return_conditional_losses_96166370]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
1__inference_Critic_readout_layer_call_fn_96166379P0?-
&?#
!?
inputs??????????
? "???????????
L__inference_Policy_layer_1_layer_call_and_return_conditional_losses_96166470l'(7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????
? ?
1__inference_Policy_layer_1_layer_call_fn_96166479_'(7?4
-?*
(?%
inputs?????????@
? " ???????????
L__inference_Policy_readout_layer_call_and_return_conditional_losses_96166350]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
1__inference_Policy_readout_layer_call_fn_96166359P0?-
&?#
!?
inputs??????????
? "???????????
#__inference__wrapped_model_96165872? !"#$%&'()*8?5
.?+
)?&
input_1?????????
? "M?J

policy?
policy
+
value_estimate?
value_estimate?
N__inference_actor_critic_363_layer_call_and_return_conditional_losses_96166124? !"#$%&'()*8?5
.?+
)?&
input_1?????????
? "[?X
Q?N

policy?
0/policy
-
value_estimate?
0/value_estimate
? ?
3__inference_actor_critic_363_layer_call_fn_96166164? !"#$%&'()*8?5
.?+
)?&
input_1?????????
? "M?J

policy?
policy
+
value_estimate?
value_estimate?
__inference_call_96166272? !"#$%&'()*3?0
)?&
$?!
observation
? "M?J

policy?
policy
)
value_estimate?
value_estimate ?
__inference_call_96166339? !"#$%&'()*<?9
2?/
-?*
observation?????????
? "M?J

policy?
policy
+
value_estimate?
value_estimate?
I__inference_flatten_726_layer_call_and_return_conditional_losses_96166485a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
.__inference_flatten_726_layer_call_fn_96166490T7?4
-?*
(?%
inputs?????????
? "????????????
I__inference_flatten_727_layer_call_and_return_conditional_losses_96166516a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
.__inference_flatten_727_layer_call_fn_96166521T7?4
-?*
(?%
inputs?????????
? "????????????
F__inference_readin_1_layer_call_and_return_conditional_losses_96166390l 7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
+__inference_readin_1_layer_call_fn_96166399_ 7?4
-?*
(?%
inputs?????????
? " ??????????@?
F__inference_readin_2_layer_call_and_return_conditional_losses_96166410l!"7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
+__inference_readin_2_layer_call_fn_96166419_!"7?4
-?*
(?%
inputs?????????@
? " ??????????@?
F__inference_readin_3_layer_call_and_return_conditional_losses_96166430l#$7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
+__inference_readin_3_layer_call_fn_96166439_#$7?4
-?*
(?%
inputs?????????@
? " ??????????@?
F__inference_readin_4_layer_call_and_return_conditional_losses_96166450l%&7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
+__inference_readin_4_layer_call_fn_96166459_%&7?4
-?*
(?%
inputs?????????@
? " ??????????@?
&__inference_signature_wrapper_96166205? !"#$%&'()*C?@
? 
9?6
4
input_1)?&
input_1?????????"M?J

policy?
policy
+
value_estimate?
value_estimate