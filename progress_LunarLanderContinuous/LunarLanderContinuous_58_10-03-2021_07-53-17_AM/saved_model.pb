²¡
®
.
Abs
x"T
y"T"
Ttype:

2	
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8	

Policy_mu_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namePolicy_mu_readout/kernel

,Policy_mu_readout/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_readout/kernel*
_output_shapes

: *
dtype0

Policy_mu_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namePolicy_mu_readout/bias
}
*Policy_mu_readout/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_readout/bias*
_output_shapes
:*
dtype0

Policy_sigma_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_namePolicy_sigma_readout/kernel

/Policy_sigma_readout/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_readout/kernel*
_output_shapes

: *
dtype0

Policy_sigma_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namePolicy_sigma_readout/bias

-Policy_sigma_readout/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_readout/bias*
_output_shapes
:*
dtype0

Value_readout/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameValue_readout/kernel
}
(Value_readout/kernel/Read/ReadVariableOpReadVariableOpValue_readout/kernel*
_output_shapes

: *
dtype0
|
Value_readout/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameValue_readout/bias
u
&Value_readout/bias/Read/ReadVariableOpReadVariableOpValue_readout/bias*
_output_shapes
:*
dtype0

Policy_mu_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_namePolicy_mu_0/kernel
y
&Policy_mu_0/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_0/kernel*
_output_shapes

: *
dtype0
x
Policy_mu_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_0/bias
q
$Policy_mu_0/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_0/bias*
_output_shapes
: *
dtype0

Policy_mu_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *#
shared_namePolicy_mu_1/kernel
y
&Policy_mu_1/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_1/kernel*
_output_shapes

:  *
dtype0
x
Policy_mu_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_1/bias
q
$Policy_mu_1/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_1/bias*
_output_shapes
: *
dtype0

Policy_mu_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *#
shared_namePolicy_mu_2/kernel
y
&Policy_mu_2/kernel/Read/ReadVariableOpReadVariableOpPolicy_mu_2/kernel*
_output_shapes

:  *
dtype0
x
Policy_mu_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namePolicy_mu_2/bias
q
$Policy_mu_2/bias/Read/ReadVariableOpReadVariableOpPolicy_mu_2/bias*
_output_shapes
: *
dtype0

Policy_sigma_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_namePolicy_sigma_0/kernel

)Policy_sigma_0/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_0/kernel*
_output_shapes

: *
dtype0
~
Policy_sigma_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_0/bias
w
'Policy_sigma_0/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_0/bias*
_output_shapes
: *
dtype0

Policy_sigma_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_namePolicy_sigma_1/kernel

)Policy_sigma_1/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_1/kernel*
_output_shapes

:  *
dtype0
~
Policy_sigma_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_1/bias
w
'Policy_sigma_1/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_1/bias*
_output_shapes
: *
dtype0

Policy_sigma_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_namePolicy_sigma_2/kernel

)Policy_sigma_2/kernel/Read/ReadVariableOpReadVariableOpPolicy_sigma_2/kernel*
_output_shapes

:  *
dtype0
~
Policy_sigma_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namePolicy_sigma_2/bias
w
'Policy_sigma_2/bias/Read/ReadVariableOpReadVariableOpPolicy_sigma_2/bias*
_output_shapes
: *
dtype0

Value_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameValue_layer_0/kernel
}
(Value_layer_0/kernel/Read/ReadVariableOpReadVariableOpValue_layer_0/kernel*
_output_shapes

: *
dtype0
|
Value_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_0/bias
u
&Value_layer_0/bias/Read/ReadVariableOpReadVariableOpValue_layer_0/bias*
_output_shapes
: *
dtype0

Value_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *%
shared_nameValue_layer_1/kernel
}
(Value_layer_1/kernel/Read/ReadVariableOpReadVariableOpValue_layer_1/kernel*
_output_shapes

:  *
dtype0
|
Value_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_1/bias
u
&Value_layer_1/bias/Read/ReadVariableOpReadVariableOpValue_layer_1/bias*
_output_shapes
: *
dtype0

Value_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *%
shared_nameValue_layer_2/kernel
}
(Value_layer_2/kernel/Read/ReadVariableOpReadVariableOpValue_layer_2/kernel*
_output_shapes

:  *
dtype0
|
Value_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameValue_layer_2/bias
u
&Value_layer_2/bias/Read/ReadVariableOpReadVariableOpValue_layer_2/bias*
_output_shapes
: *
dtype0

NoOpNoOp
å7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 7
value7B7 B7
È
mu_layer

readout_mu
sigma_layer
readout_sigma
value_layer
readout_value
trainable_variables
	variables
	regularization_losses

	keras_api

signatures

0
1
2
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
 2
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
¶
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23
¶
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23
 
­
9non_trainable_variables
trainable_variables
:metrics

;layers
	variables
<layer_metrics
=layer_regularization_losses
	regularization_losses
 
h

'kernel
(bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

)kernel
*bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
h

+kernel
,bias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
ZX
VARIABLE_VALUEPolicy_mu_readout/kernel,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_readout/bias*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables
Lmetrics
	variables
Mlayer_metrics

Nlayers
regularization_losses
h

-kernel
.bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

/kernel
0bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
h

1kernel
2bias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
`^
VARIABLE_VALUEPolicy_sigma_readout/kernel/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_readout/bias-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
[non_trainable_variables
\layer_regularization_losses
trainable_variables
]metrics
	variables
^layer_metrics

_layers
regularization_losses
h

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
h

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
h

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
YW
VARIABLE_VALUEValue_readout/kernel/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEValue_readout/bias-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
­
lnon_trainable_variables
mlayer_regularization_losses
#trainable_variables
nmetrics
$	variables
olayer_metrics

players
%regularization_losses
XV
VARIABLE_VALUEPolicy_mu_0/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_0/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEPolicy_mu_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEPolicy_mu_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEPolicy_mu_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEPolicy_sigma_0/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEPolicy_sigma_0/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPolicy_sigma_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEPolicy_sigma_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEPolicy_sigma_2/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_0/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_0/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_1/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_1/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEValue_layer_2/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEValue_layer_2/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
8
9
 10
11
 
 

'0
(1

'0
(1
 
­
qnon_trainable_variables
rlayer_regularization_losses
>trainable_variables
smetrics
?	variables
tlayer_metrics

ulayers
@regularization_losses

)0
*1

)0
*1
 
­
vnon_trainable_variables
wlayer_regularization_losses
Btrainable_variables
xmetrics
C	variables
ylayer_metrics

zlayers
Dregularization_losses

+0
,1

+0
,1
 
­
{non_trainable_variables
|layer_regularization_losses
Ftrainable_variables
}metrics
G	variables
~layer_metrics

layers
Hregularization_losses
 
 
 
 
 

-0
.1

-0
.1
 
²
non_trainable_variables
 layer_regularization_losses
Otrainable_variables
metrics
P	variables
layer_metrics
layers
Qregularization_losses

/0
01

/0
01
 
²
non_trainable_variables
 layer_regularization_losses
Strainable_variables
metrics
T	variables
layer_metrics
layers
Uregularization_losses

10
21

10
21
 
²
non_trainable_variables
 layer_regularization_losses
Wtrainable_variables
metrics
X	variables
layer_metrics
layers
Yregularization_losses
 
 
 
 
 

30
41

30
41
 
²
non_trainable_variables
 layer_regularization_losses
`trainable_variables
metrics
a	variables
layer_metrics
layers
bregularization_losses

50
61

50
61
 
²
non_trainable_variables
 layer_regularization_losses
dtrainable_variables
metrics
e	variables
layer_metrics
layers
fregularization_losses

70
81

70
81
 
²
non_trainable_variables
 layer_regularization_losses
htrainable_variables
metrics
i	variables
layer_metrics
layers
jregularization_losses
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
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ç
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Policy_mu_0/kernelPolicy_mu_0/biasPolicy_mu_1/kernelPolicy_mu_1/biasPolicy_mu_2/kernelPolicy_mu_2/biasPolicy_sigma_0/kernelPolicy_sigma_0/biasPolicy_sigma_1/kernelPolicy_sigma_1/biasPolicy_sigma_2/kernelPolicy_sigma_2/biasValue_layer_0/kernelValue_layer_0/biasValue_layer_1/kernelValue_layer_1/biasValue_layer_2/kernelValue_layer_2/biasPolicy_mu_readout/kernelPolicy_mu_readout/biasPolicy_sigma_readout/kernelPolicy_sigma_readout/biasValue_readout/kernelValue_readout/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:::*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_14718083
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,Policy_mu_readout/kernel/Read/ReadVariableOp*Policy_mu_readout/bias/Read/ReadVariableOp/Policy_sigma_readout/kernel/Read/ReadVariableOp-Policy_sigma_readout/bias/Read/ReadVariableOp(Value_readout/kernel/Read/ReadVariableOp&Value_readout/bias/Read/ReadVariableOp&Policy_mu_0/kernel/Read/ReadVariableOp$Policy_mu_0/bias/Read/ReadVariableOp&Policy_mu_1/kernel/Read/ReadVariableOp$Policy_mu_1/bias/Read/ReadVariableOp&Policy_mu_2/kernel/Read/ReadVariableOp$Policy_mu_2/bias/Read/ReadVariableOp)Policy_sigma_0/kernel/Read/ReadVariableOp'Policy_sigma_0/bias/Read/ReadVariableOp)Policy_sigma_1/kernel/Read/ReadVariableOp'Policy_sigma_1/bias/Read/ReadVariableOp)Policy_sigma_2/kernel/Read/ReadVariableOp'Policy_sigma_2/bias/Read/ReadVariableOp(Value_layer_0/kernel/Read/ReadVariableOp&Value_layer_0/bias/Read/ReadVariableOp(Value_layer_1/kernel/Read/ReadVariableOp&Value_layer_1/bias/Read/ReadVariableOp(Value_layer_2/kernel/Read/ReadVariableOp&Value_layer_2/bias/Read/ReadVariableOpConst*%
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_14718599
¸
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamePolicy_mu_readout/kernelPolicy_mu_readout/biasPolicy_sigma_readout/kernelPolicy_sigma_readout/biasValue_readout/kernelValue_readout/biasPolicy_mu_0/kernelPolicy_mu_0/biasPolicy_mu_1/kernelPolicy_mu_1/biasPolicy_mu_2/kernelPolicy_mu_2/biasPolicy_sigma_0/kernelPolicy_sigma_0/biasPolicy_sigma_1/kernelPolicy_sigma_1/biasPolicy_sigma_2/kernelPolicy_sigma_2/biasValue_layer_0/kernelValue_layer_0/biasValue_layer_1/kernelValue_layer_1/biasValue_layer_2/kernelValue_layer_2/bias*$
Tin
2*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_14718681ÿ
ë

0__inference_Value_layer_1_layer_call_fn_14718482

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_147178382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó	
â
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_14717649

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
å
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_14718413

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õ	
ä
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_14717838

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õ	
ä
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_14718473

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ú

#__inference__wrapped_model_14717634
input_1
a2c_120_14717580
a2c_120_14717582
a2c_120_14717584
a2c_120_14717586
a2c_120_14717588
a2c_120_14717590
a2c_120_14717592
a2c_120_14717594
a2c_120_14717596
a2c_120_14717598
a2c_120_14717600
a2c_120_14717602
a2c_120_14717604
a2c_120_14717606
a2c_120_14717608
a2c_120_14717610
a2c_120_14717612
a2c_120_14717614
a2c_120_14717616
a2c_120_14717618
a2c_120_14717620
a2c_120_14717622
a2c_120_14717624
a2c_120_14717626
identity

identity_1

identity_2¢a2c_120/StatefulPartitionedCall
a2c_120/StatefulPartitionedCallStatefulPartitionedCallinput_1a2c_120_14717580a2c_120_14717582a2c_120_14717584a2c_120_14717586a2c_120_14717588a2c_120_14717590a2c_120_14717592a2c_120_14717594a2c_120_14717596a2c_120_14717598a2c_120_14717600a2c_120_14717602a2c_120_14717604a2c_120_14717606a2c_120_14717608a2c_120_14717610a2c_120_14717612a2c_120_14717614a2c_120_14717616a2c_120_14717618a2c_120_14717620a2c_120_14717622a2c_120_14717624a2c_120_14717626*$
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:::*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_call_147175792!
a2c_120/StatefulPartitionedCall
IdentityIdentity(a2c_120/StatefulPartitionedCall:output:0 ^a2c_120/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity(a2c_120/StatefulPartitionedCall:output:1 ^a2c_120/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1

Identity_2Identity(a2c_120/StatefulPartitionedCall:output:2 ^a2c_120/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2B
a2c_120/StatefulPartitionedCalla2c_120/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ó	
â
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_14718373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö	
å
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_14717757

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õ	
ä
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_14717865

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
õ	
ä
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_14718453

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ä
K__inference_Value_readout_layer_call_and_return_conditional_losses_14717946

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô
ó
&__inference_signature_wrapper_14718083
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:::*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_147176342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
 	
ë
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_14718294

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

1__inference_Policy_sigma_2_layer_call_fn_14718442

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_147177842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ËQ


E__inference_a2c_120_layer_call_and_return_conditional_losses_14717966
input_1
policy_mu_0_14717660
policy_mu_0_14717662
policy_mu_1_14717687
policy_mu_1_14717689
policy_mu_2_14717714
policy_mu_2_14717716
policy_sigma_0_14717741
policy_sigma_0_14717743
policy_sigma_1_14717768
policy_sigma_1_14717770
policy_sigma_2_14717795
policy_sigma_2_14717797
value_layer_0_14717822
value_layer_0_14717824
value_layer_1_14717849
value_layer_1_14717851
value_layer_2_14717876
value_layer_2_14717878
policy_mu_readout_14717902
policy_mu_readout_14717904!
policy_sigma_readout_14717929!
policy_sigma_readout_14717931
value_readout_14717957
value_readout_14717959
identity

identity_1

identity_2¢#Policy_mu_0/StatefulPartitionedCall¢#Policy_mu_1/StatefulPartitionedCall¢#Policy_mu_2/StatefulPartitionedCall¢)Policy_mu_readout/StatefulPartitionedCall¢&Policy_sigma_0/StatefulPartitionedCall¢&Policy_sigma_1/StatefulPartitionedCall¢&Policy_sigma_2/StatefulPartitionedCall¢,Policy_sigma_readout/StatefulPartitionedCall¢%Value_layer_0/StatefulPartitionedCall¢%Value_layer_1/StatefulPartitionedCall¢%Value_layer_2/StatefulPartitionedCall¢%Value_readout/StatefulPartitionedCallª
#Policy_mu_0/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_mu_0_14717660policy_mu_0_14717662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_147176492%
#Policy_mu_0/StatefulPartitionedCallÏ
#Policy_mu_1/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_0/StatefulPartitionedCall:output:0policy_mu_1_14717687policy_mu_1_14717689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_147176762%
#Policy_mu_1/StatefulPartitionedCallÏ
#Policy_mu_2/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_1/StatefulPartitionedCall:output:0policy_mu_2_14717714policy_mu_2_14717716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_147177032%
#Policy_mu_2/StatefulPartitionedCall¹
&Policy_sigma_0/StatefulPartitionedCallStatefulPartitionedCallinput_1policy_sigma_0_14717741policy_sigma_0_14717743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_147177302(
&Policy_sigma_0/StatefulPartitionedCallá
&Policy_sigma_1/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_0/StatefulPartitionedCall:output:0policy_sigma_1_14717768policy_sigma_1_14717770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_147177572(
&Policy_sigma_1/StatefulPartitionedCallá
&Policy_sigma_2/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_1/StatefulPartitionedCall:output:0policy_sigma_2_14717795policy_sigma_2_14717797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_147177842(
&Policy_sigma_2/StatefulPartitionedCall´
%Value_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_1value_layer_0_14717822value_layer_0_14717824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_147178112'
%Value_layer_0/StatefulPartitionedCallÛ
%Value_layer_1/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_0/StatefulPartitionedCall:output:0value_layer_1_14717849value_layer_1_14717851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_147178382'
%Value_layer_1/StatefulPartitionedCallÛ
%Value_layer_2/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_1/StatefulPartitionedCall:output:0value_layer_2_14717876value_layer_2_14717878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_147178652'
%Value_layer_2/StatefulPartitionedCallí
)Policy_mu_readout/StatefulPartitionedCallStatefulPartitionedCall,Policy_mu_2/StatefulPartitionedCall:output:0policy_mu_readout_14717902policy_mu_readout_14717904*
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
GPU 2J 8 *X
fSRQ
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_147178912+
)Policy_mu_readout/StatefulPartitionedCallt
SqueezeSqueeze2Policy_mu_readout/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2	
Squeezeÿ
,Policy_sigma_readout/StatefulPartitionedCallStatefulPartitionedCall/Policy_sigma_2/StatefulPartitionedCall:output:0policy_sigma_readout_14717929policy_sigma_readout_14717931*
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
GPU 2J 8 *[
fVRT
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_147179182.
,Policy_sigma_readout/StatefulPartitionedCallz
AbsAbs5Policy_sigma_readout/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1Û
%Value_readout/StatefulPartitionedCallStatefulPartitionedCall.Value_layer_2/StatefulPartitionedCall:output:0value_readout_14717957value_readout_14717959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_readout_layer_call_and_return_conditional_losses_147179462'
%Value_readout/StatefulPartitionedCallt
	Squeeze_2Squeeze.Value_readout/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
	Squeeze_2½
IdentityIdentitySqueeze:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

IdentityÃ

Identity_1IdentitySqueeze_1:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1Ã

Identity_2IdentitySqueeze_2:output:0$^Policy_mu_0/StatefulPartitionedCall$^Policy_mu_1/StatefulPartitionedCall$^Policy_mu_2/StatefulPartitionedCall*^Policy_mu_readout/StatefulPartitionedCall'^Policy_sigma_0/StatefulPartitionedCall'^Policy_sigma_1/StatefulPartitionedCall'^Policy_sigma_2/StatefulPartitionedCall-^Policy_sigma_readout/StatefulPartitionedCall&^Value_layer_0/StatefulPartitionedCall&^Value_layer_1/StatefulPartitionedCall&^Value_layer_2/StatefulPartitionedCall&^Value_readout/StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2J
#Policy_mu_0/StatefulPartitionedCall#Policy_mu_0/StatefulPartitionedCall2J
#Policy_mu_1/StatefulPartitionedCall#Policy_mu_1/StatefulPartitionedCall2J
#Policy_mu_2/StatefulPartitionedCall#Policy_mu_2/StatefulPartitionedCall2V
)Policy_mu_readout/StatefulPartitionedCall)Policy_mu_readout/StatefulPartitionedCall2P
&Policy_sigma_0/StatefulPartitionedCall&Policy_sigma_0/StatefulPartitionedCall2P
&Policy_sigma_1/StatefulPartitionedCall&Policy_sigma_1/StatefulPartitionedCall2P
&Policy_sigma_2/StatefulPartitionedCall&Policy_sigma_2/StatefulPartitionedCall2\
,Policy_sigma_readout/StatefulPartitionedCall,Policy_sigma_readout/StatefulPartitionedCall2N
%Value_layer_0/StatefulPartitionedCall%Value_layer_0/StatefulPartitionedCall2N
%Value_layer_1/StatefulPartitionedCall%Value_layer_1/StatefulPartitionedCall2N
%Value_layer_2/StatefulPartitionedCall%Value_layer_2/StatefulPartitionedCall2N
%Value_readout/StatefulPartitionedCall%Value_readout/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Æ
Õ
__inference_call_14718265
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2¢"Policy_mu_0/BiasAdd/ReadVariableOp¢!Policy_mu_0/MatMul/ReadVariableOp¢"Policy_mu_1/BiasAdd/ReadVariableOp¢!Policy_mu_1/MatMul/ReadVariableOp¢"Policy_mu_2/BiasAdd/ReadVariableOp¢!Policy_mu_2/MatMul/ReadVariableOp¢(Policy_mu_readout/BiasAdd/ReadVariableOp¢'Policy_mu_readout/MatMul/ReadVariableOp¢%Policy_sigma_0/BiasAdd/ReadVariableOp¢$Policy_sigma_0/MatMul/ReadVariableOp¢%Policy_sigma_1/BiasAdd/ReadVariableOp¢$Policy_sigma_1/MatMul/ReadVariableOp¢%Policy_sigma_2/BiasAdd/ReadVariableOp¢$Policy_sigma_2/MatMul/ReadVariableOp¢+Policy_sigma_readout/BiasAdd/ReadVariableOp¢*Policy_sigma_readout/MatMul/ReadVariableOp¢$Value_layer_0/BiasAdd/ReadVariableOp¢#Value_layer_0/MatMul/ReadVariableOp¢$Value_layer_1/BiasAdd/ReadVariableOp¢#Value_layer_1/MatMul/ReadVariableOp¢$Value_layer_2/BiasAdd/ReadVariableOp¢#Value_layer_2/MatMul/ReadVariableOp¢$Value_readout/BiasAdd/ReadVariableOp¢#Value_readout/MatMul/ReadVariableOp±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp±
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/BiasAdd|
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¯
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp±
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/BiasAdd|
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¯
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp±
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/BiasAdd|
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp¥
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp½
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/BiasAdd
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp»
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp½
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/BiasAdd
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp»
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp½
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/BiasAdd
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp¢
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp¹
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/BiasAdd
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp·
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp¹
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/BiasAdd
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp·
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp¹
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/BiasAdd
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOpÁ
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÉ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/BiasAddd
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÍ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÕ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/BiasAddj
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp·
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp¹
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/BiasAddd
	Squeeze_2SqueezeValue_readout/BiasAdd:output:0*
T0*
_output_shapes
:2
	Squeeze_2
IdentityIdentitySqueeze:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity

Identity_1IdentitySqueeze_1:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1

Identity_2IdentitySqueeze_2:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2H
"Policy_mu_0/BiasAdd/ReadVariableOp"Policy_mu_0/BiasAdd/ReadVariableOp2F
!Policy_mu_0/MatMul/ReadVariableOp!Policy_mu_0/MatMul/ReadVariableOp2H
"Policy_mu_1/BiasAdd/ReadVariableOp"Policy_mu_1/BiasAdd/ReadVariableOp2F
!Policy_mu_1/MatMul/ReadVariableOp!Policy_mu_1/MatMul/ReadVariableOp2H
"Policy_mu_2/BiasAdd/ReadVariableOp"Policy_mu_2/BiasAdd/ReadVariableOp2F
!Policy_mu_2/MatMul/ReadVariableOp!Policy_mu_2/MatMul/ReadVariableOp2T
(Policy_mu_readout/BiasAdd/ReadVariableOp(Policy_mu_readout/BiasAdd/ReadVariableOp2R
'Policy_mu_readout/MatMul/ReadVariableOp'Policy_mu_readout/MatMul/ReadVariableOp2N
%Policy_sigma_0/BiasAdd/ReadVariableOp%Policy_sigma_0/BiasAdd/ReadVariableOp2L
$Policy_sigma_0/MatMul/ReadVariableOp$Policy_sigma_0/MatMul/ReadVariableOp2N
%Policy_sigma_1/BiasAdd/ReadVariableOp%Policy_sigma_1/BiasAdd/ReadVariableOp2L
$Policy_sigma_1/MatMul/ReadVariableOp$Policy_sigma_1/MatMul/ReadVariableOp2N
%Policy_sigma_2/BiasAdd/ReadVariableOp%Policy_sigma_2/BiasAdd/ReadVariableOp2L
$Policy_sigma_2/MatMul/ReadVariableOp$Policy_sigma_2/MatMul/ReadVariableOp2Z
+Policy_sigma_readout/BiasAdd/ReadVariableOp+Policy_sigma_readout/BiasAdd/ReadVariableOp2X
*Policy_sigma_readout/MatMul/ReadVariableOp*Policy_sigma_readout/MatMul/ReadVariableOp2L
$Value_layer_0/BiasAdd/ReadVariableOp$Value_layer_0/BiasAdd/ReadVariableOp2J
#Value_layer_0/MatMul/ReadVariableOp#Value_layer_0/MatMul/ReadVariableOp2L
$Value_layer_1/BiasAdd/ReadVariableOp$Value_layer_1/BiasAdd/ReadVariableOp2J
#Value_layer_1/MatMul/ReadVariableOp#Value_layer_1/MatMul/ReadVariableOp2L
$Value_layer_2/BiasAdd/ReadVariableOp$Value_layer_2/BiasAdd/ReadVariableOp2J
#Value_layer_2/MatMul/ReadVariableOp#Value_layer_2/MatMul/ReadVariableOp2L
$Value_readout/BiasAdd/ReadVariableOp$Value_readout/BiasAdd/ReadVariableOp2J
#Value_readout/MatMul/ReadVariableOp#Value_readout/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
õ	
ä
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_14718493

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó	
â
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_14717676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó	
â
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_14718333

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
è
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_14717891

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

÷
*__inference_a2c_120_layer_call_fn_14718024
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 * 
_output_shapes
:::*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_a2c_120_layer_call_and_return_conditional_losses_147179662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Æ
Õ
__inference_call_14717579
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2¢"Policy_mu_0/BiasAdd/ReadVariableOp¢!Policy_mu_0/MatMul/ReadVariableOp¢"Policy_mu_1/BiasAdd/ReadVariableOp¢!Policy_mu_1/MatMul/ReadVariableOp¢"Policy_mu_2/BiasAdd/ReadVariableOp¢!Policy_mu_2/MatMul/ReadVariableOp¢(Policy_mu_readout/BiasAdd/ReadVariableOp¢'Policy_mu_readout/MatMul/ReadVariableOp¢%Policy_sigma_0/BiasAdd/ReadVariableOp¢$Policy_sigma_0/MatMul/ReadVariableOp¢%Policy_sigma_1/BiasAdd/ReadVariableOp¢$Policy_sigma_1/MatMul/ReadVariableOp¢%Policy_sigma_2/BiasAdd/ReadVariableOp¢$Policy_sigma_2/MatMul/ReadVariableOp¢+Policy_sigma_readout/BiasAdd/ReadVariableOp¢*Policy_sigma_readout/MatMul/ReadVariableOp¢$Value_layer_0/BiasAdd/ReadVariableOp¢#Value_layer_0/MatMul/ReadVariableOp¢$Value_layer_1/BiasAdd/ReadVariableOp¢#Value_layer_1/MatMul/ReadVariableOp¢$Value_layer_2/BiasAdd/ReadVariableOp¢#Value_layer_2/MatMul/ReadVariableOp¢$Value_readout/BiasAdd/ReadVariableOp¢#Value_readout/MatMul/ReadVariableOp±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp±
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/BiasAdd|
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¯
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp±
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/BiasAdd|
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¯
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp±
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/BiasAdd|
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp¥
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp½
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/BiasAdd
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp»
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp½
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/BiasAdd
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp»
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp½
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/BiasAdd
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp¢
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp¹
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/BiasAdd
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp·
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp¹
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/BiasAdd
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp·
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp¹
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/BiasAdd
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOpÁ
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÉ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_mu_readout/BiasAddd
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÍ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÕ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Policy_sigma_readout/BiasAddj
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
AbsM
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp·
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp¹
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Value_readout/BiasAddd
	Squeeze_2SqueezeValue_readout/BiasAdd:output:0*
T0*
_output_shapes
:2
	Squeeze_2
IdentityIdentitySqueeze:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity

Identity_1IdentitySqueeze_1:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1

Identity_2IdentitySqueeze_2:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapesu
s:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::2H
"Policy_mu_0/BiasAdd/ReadVariableOp"Policy_mu_0/BiasAdd/ReadVariableOp2F
!Policy_mu_0/MatMul/ReadVariableOp!Policy_mu_0/MatMul/ReadVariableOp2H
"Policy_mu_1/BiasAdd/ReadVariableOp"Policy_mu_1/BiasAdd/ReadVariableOp2F
!Policy_mu_1/MatMul/ReadVariableOp!Policy_mu_1/MatMul/ReadVariableOp2H
"Policy_mu_2/BiasAdd/ReadVariableOp"Policy_mu_2/BiasAdd/ReadVariableOp2F
!Policy_mu_2/MatMul/ReadVariableOp!Policy_mu_2/MatMul/ReadVariableOp2T
(Policy_mu_readout/BiasAdd/ReadVariableOp(Policy_mu_readout/BiasAdd/ReadVariableOp2R
'Policy_mu_readout/MatMul/ReadVariableOp'Policy_mu_readout/MatMul/ReadVariableOp2N
%Policy_sigma_0/BiasAdd/ReadVariableOp%Policy_sigma_0/BiasAdd/ReadVariableOp2L
$Policy_sigma_0/MatMul/ReadVariableOp$Policy_sigma_0/MatMul/ReadVariableOp2N
%Policy_sigma_1/BiasAdd/ReadVariableOp%Policy_sigma_1/BiasAdd/ReadVariableOp2L
$Policy_sigma_1/MatMul/ReadVariableOp$Policy_sigma_1/MatMul/ReadVariableOp2N
%Policy_sigma_2/BiasAdd/ReadVariableOp%Policy_sigma_2/BiasAdd/ReadVariableOp2L
$Policy_sigma_2/MatMul/ReadVariableOp$Policy_sigma_2/MatMul/ReadVariableOp2Z
+Policy_sigma_readout/BiasAdd/ReadVariableOp+Policy_sigma_readout/BiasAdd/ReadVariableOp2X
*Policy_sigma_readout/MatMul/ReadVariableOp*Policy_sigma_readout/MatMul/ReadVariableOp2L
$Value_layer_0/BiasAdd/ReadVariableOp$Value_layer_0/BiasAdd/ReadVariableOp2J
#Value_layer_0/MatMul/ReadVariableOp#Value_layer_0/MatMul/ReadVariableOp2L
$Value_layer_1/BiasAdd/ReadVariableOp$Value_layer_1/BiasAdd/ReadVariableOp2J
#Value_layer_1/MatMul/ReadVariableOp#Value_layer_1/MatMul/ReadVariableOp2L
$Value_layer_2/BiasAdd/ReadVariableOp$Value_layer_2/BiasAdd/ReadVariableOp2J
#Value_layer_2/MatMul/ReadVariableOp#Value_layer_2/MatMul/ReadVariableOp2L
$Value_readout/BiasAdd/ReadVariableOp$Value_readout/BiasAdd/ReadVariableOp2J
#Value_readout/MatMul/ReadVariableOp#Value_readout/MatMul/ReadVariableOp:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_state
ö	
å
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_14718393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_0_layer_call_fn_14718342

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_147176492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µe

$__inference__traced_restore_14718681
file_prefix-
)assignvariableop_policy_mu_readout_kernel-
)assignvariableop_1_policy_mu_readout_bias2
.assignvariableop_2_policy_sigma_readout_kernel0
,assignvariableop_3_policy_sigma_readout_bias+
'assignvariableop_4_value_readout_kernel)
%assignvariableop_5_value_readout_bias)
%assignvariableop_6_policy_mu_0_kernel'
#assignvariableop_7_policy_mu_0_bias)
%assignvariableop_8_policy_mu_1_kernel'
#assignvariableop_9_policy_mu_1_bias*
&assignvariableop_10_policy_mu_2_kernel(
$assignvariableop_11_policy_mu_2_bias-
)assignvariableop_12_policy_sigma_0_kernel+
'assignvariableop_13_policy_sigma_0_bias-
)assignvariableop_14_policy_sigma_1_kernel+
'assignvariableop_15_policy_sigma_1_bias-
)assignvariableop_16_policy_sigma_2_kernel+
'assignvariableop_17_policy_sigma_2_bias,
(assignvariableop_18_value_layer_0_kernel*
&assignvariableop_19_value_layer_0_bias,
(assignvariableop_20_value_layer_1_kernel*
&assignvariableop_21_value_layer_1_bias,
(assignvariableop_22_value_layer_2_kernel*
&assignvariableop_23_value_layer_2_bias
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ï

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û	
valueÑ	BÎ	B,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÀ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¨
AssignVariableOpAssignVariableOp)assignvariableop_policy_mu_readout_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1®
AssignVariableOp_1AssignVariableOp)assignvariableop_1_policy_mu_readout_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_policy_sigma_readout_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_policy_sigma_readout_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_value_readout_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_value_readout_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_policy_mu_0_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¨
AssignVariableOp_7AssignVariableOp#assignvariableop_7_policy_mu_0_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_policy_mu_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_policy_mu_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_policy_mu_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¬
AssignVariableOp_11AssignVariableOp$assignvariableop_11_policy_mu_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_policy_sigma_0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¯
AssignVariableOp_13AssignVariableOp'assignvariableop_13_policy_sigma_0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14±
AssignVariableOp_14AssignVariableOp)assignvariableop_14_policy_sigma_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¯
AssignVariableOp_15AssignVariableOp'assignvariableop_15_policy_sigma_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_policy_sigma_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¯
AssignVariableOp_17AssignVariableOp'assignvariableop_17_policy_sigma_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_value_layer_0_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp&assignvariableop_19_value_layer_0_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_value_layer_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp&assignvariableop_21_value_layer_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_value_layer_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp&assignvariableop_23_value_layer_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpî
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24á
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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
 	
ë
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_14717918

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ù

7__inference_Policy_sigma_readout_layer_call_fn_14718303

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
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
GPU 2J 8 *[
fVRT
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_147179182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í

1__inference_Policy_sigma_1_layer_call_fn_14718422

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_147177572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_2_layer_call_fn_14718382

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_147177032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿ
Õ
__inference_call_14718174
input_state.
*policy_mu_0_matmul_readvariableop_resource/
+policy_mu_0_biasadd_readvariableop_resource.
*policy_mu_1_matmul_readvariableop_resource/
+policy_mu_1_biasadd_readvariableop_resource.
*policy_mu_2_matmul_readvariableop_resource/
+policy_mu_2_biasadd_readvariableop_resource1
-policy_sigma_0_matmul_readvariableop_resource2
.policy_sigma_0_biasadd_readvariableop_resource1
-policy_sigma_1_matmul_readvariableop_resource2
.policy_sigma_1_biasadd_readvariableop_resource1
-policy_sigma_2_matmul_readvariableop_resource2
.policy_sigma_2_biasadd_readvariableop_resource0
,value_layer_0_matmul_readvariableop_resource1
-value_layer_0_biasadd_readvariableop_resource0
,value_layer_1_matmul_readvariableop_resource1
-value_layer_1_biasadd_readvariableop_resource0
,value_layer_2_matmul_readvariableop_resource1
-value_layer_2_biasadd_readvariableop_resource4
0policy_mu_readout_matmul_readvariableop_resource5
1policy_mu_readout_biasadd_readvariableop_resource7
3policy_sigma_readout_matmul_readvariableop_resource8
4policy_sigma_readout_biasadd_readvariableop_resource0
,value_readout_matmul_readvariableop_resource1
-value_readout_biasadd_readvariableop_resource
identity

identity_1

identity_2¢"Policy_mu_0/BiasAdd/ReadVariableOp¢!Policy_mu_0/MatMul/ReadVariableOp¢"Policy_mu_1/BiasAdd/ReadVariableOp¢!Policy_mu_1/MatMul/ReadVariableOp¢"Policy_mu_2/BiasAdd/ReadVariableOp¢!Policy_mu_2/MatMul/ReadVariableOp¢(Policy_mu_readout/BiasAdd/ReadVariableOp¢'Policy_mu_readout/MatMul/ReadVariableOp¢%Policy_sigma_0/BiasAdd/ReadVariableOp¢$Policy_sigma_0/MatMul/ReadVariableOp¢%Policy_sigma_1/BiasAdd/ReadVariableOp¢$Policy_sigma_1/MatMul/ReadVariableOp¢%Policy_sigma_2/BiasAdd/ReadVariableOp¢$Policy_sigma_2/MatMul/ReadVariableOp¢+Policy_sigma_readout/BiasAdd/ReadVariableOp¢*Policy_sigma_readout/MatMul/ReadVariableOp¢$Value_layer_0/BiasAdd/ReadVariableOp¢#Value_layer_0/MatMul/ReadVariableOp¢$Value_layer_1/BiasAdd/ReadVariableOp¢#Value_layer_1/MatMul/ReadVariableOp¢$Value_layer_2/BiasAdd/ReadVariableOp¢#Value_layer_2/MatMul/ReadVariableOp¢$Value_readout/BiasAdd/ReadVariableOp¢#Value_readout/MatMul/ReadVariableOp±
!Policy_mu_0/MatMul/ReadVariableOpReadVariableOp*policy_mu_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!Policy_mu_0/MatMul/ReadVariableOp
Policy_mu_0/MatMulMatMulinput_state)Policy_mu_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_0/MatMul°
"Policy_mu_0/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_0/BiasAdd/ReadVariableOp¨
Policy_mu_0/BiasAddBiasAddPolicy_mu_0/MatMul:product:0*Policy_mu_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_0/BiasAdds
Policy_mu_0/ReluReluPolicy_mu_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_0/Relu±
!Policy_mu_1/MatMul/ReadVariableOpReadVariableOp*policy_mu_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_1/MatMul/ReadVariableOp¦
Policy_mu_1/MatMulMatMulPolicy_mu_0/Relu:activations:0)Policy_mu_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_1/MatMul°
"Policy_mu_1/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_1/BiasAdd/ReadVariableOp¨
Policy_mu_1/BiasAddBiasAddPolicy_mu_1/MatMul:product:0*Policy_mu_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_1/BiasAdds
Policy_mu_1/ReluReluPolicy_mu_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_1/Relu±
!Policy_mu_2/MatMul/ReadVariableOpReadVariableOp*policy_mu_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02#
!Policy_mu_2/MatMul/ReadVariableOp¦
Policy_mu_2/MatMulMatMulPolicy_mu_1/Relu:activations:0)Policy_mu_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_2/MatMul°
"Policy_mu_2/BiasAdd/ReadVariableOpReadVariableOp+policy_mu_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"Policy_mu_2/BiasAdd/ReadVariableOp¨
Policy_mu_2/BiasAddBiasAddPolicy_mu_2/MatMul:product:0*Policy_mu_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_mu_2/BiasAdds
Policy_mu_2/ReluReluPolicy_mu_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_mu_2/Reluº
$Policy_sigma_0/MatMul/ReadVariableOpReadVariableOp-policy_sigma_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Policy_sigma_0/MatMul/ReadVariableOp
Policy_sigma_0/MatMulMatMulinput_state,Policy_sigma_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_0/MatMul¹
%Policy_sigma_0/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_0/BiasAdd/ReadVariableOp´
Policy_sigma_0/BiasAddBiasAddPolicy_sigma_0/MatMul:product:0-Policy_sigma_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_0/BiasAdd|
Policy_sigma_0/ReluReluPolicy_sigma_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_0/Reluº
$Policy_sigma_1/MatMul/ReadVariableOpReadVariableOp-policy_sigma_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_1/MatMul/ReadVariableOp²
Policy_sigma_1/MatMulMatMul!Policy_sigma_0/Relu:activations:0,Policy_sigma_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_1/MatMul¹
%Policy_sigma_1/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_1/BiasAdd/ReadVariableOp´
Policy_sigma_1/BiasAddBiasAddPolicy_sigma_1/MatMul:product:0-Policy_sigma_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_1/BiasAdd|
Policy_sigma_1/ReluReluPolicy_sigma_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_1/Reluº
$Policy_sigma_2/MatMul/ReadVariableOpReadVariableOp-policy_sigma_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02&
$Policy_sigma_2/MatMul/ReadVariableOp²
Policy_sigma_2/MatMulMatMul!Policy_sigma_1/Relu:activations:0,Policy_sigma_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_2/MatMul¹
%Policy_sigma_2/BiasAdd/ReadVariableOpReadVariableOp.policy_sigma_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%Policy_sigma_2/BiasAdd/ReadVariableOp´
Policy_sigma_2/BiasAddBiasAddPolicy_sigma_2/MatMul:product:0-Policy_sigma_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Policy_sigma_2/BiasAdd|
Policy_sigma_2/ReluReluPolicy_sigma_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Policy_sigma_2/Relu·
#Value_layer_0/MatMul/ReadVariableOpReadVariableOp,value_layer_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_layer_0/MatMul/ReadVariableOp
Value_layer_0/MatMulMatMulinput_state+Value_layer_0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_0/MatMul¶
$Value_layer_0/BiasAdd/ReadVariableOpReadVariableOp-value_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_0/BiasAdd/ReadVariableOp°
Value_layer_0/BiasAddBiasAddValue_layer_0/MatMul:product:0,Value_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_0/BiasAddy
Value_layer_0/ReluReluValue_layer_0/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_0/Relu·
#Value_layer_1/MatMul/ReadVariableOpReadVariableOp,value_layer_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_1/MatMul/ReadVariableOp®
Value_layer_1/MatMulMatMul Value_layer_0/Relu:activations:0+Value_layer_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_1/MatMul¶
$Value_layer_1/BiasAdd/ReadVariableOpReadVariableOp-value_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_1/BiasAdd/ReadVariableOp°
Value_layer_1/BiasAddBiasAddValue_layer_1/MatMul:product:0,Value_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_1/BiasAddy
Value_layer_1/ReluReluValue_layer_1/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_1/Relu·
#Value_layer_2/MatMul/ReadVariableOpReadVariableOp,value_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#Value_layer_2/MatMul/ReadVariableOp®
Value_layer_2/MatMulMatMul Value_layer_1/Relu:activations:0+Value_layer_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_2/MatMul¶
$Value_layer_2/BiasAdd/ReadVariableOpReadVariableOp-value_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$Value_layer_2/BiasAdd/ReadVariableOp°
Value_layer_2/BiasAddBiasAddValue_layer_2/MatMul:product:0,Value_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 2
Value_layer_2/BiasAddy
Value_layer_2/ReluReluValue_layer_2/BiasAdd:output:0*
T0*
_output_shapes

: 2
Value_layer_2/ReluÃ
'Policy_mu_readout/MatMul/ReadVariableOpReadVariableOp0policy_mu_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02)
'Policy_mu_readout/MatMul/ReadVariableOp¸
Policy_mu_readout/MatMulMatMulPolicy_mu_2/Relu:activations:0/Policy_mu_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_mu_readout/MatMulÂ
(Policy_mu_readout/BiasAdd/ReadVariableOpReadVariableOp1policy_mu_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Policy_mu_readout/BiasAdd/ReadVariableOpÀ
Policy_mu_readout/BiasAddBiasAdd"Policy_mu_readout/MatMul:product:00Policy_mu_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_mu_readout/BiasAddf
SqueezeSqueeze"Policy_mu_readout/BiasAdd:output:0*
T0*
_output_shapes
:2	
SqueezeÌ
*Policy_sigma_readout/MatMul/ReadVariableOpReadVariableOp3policy_sigma_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*Policy_sigma_readout/MatMul/ReadVariableOpÄ
Policy_sigma_readout/MatMulMatMul!Policy_sigma_2/Relu:activations:02Policy_sigma_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_sigma_readout/MatMulË
+Policy_sigma_readout/BiasAdd/ReadVariableOpReadVariableOp4policy_sigma_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+Policy_sigma_readout/BiasAdd/ReadVariableOpÌ
Policy_sigma_readout/BiasAddBiasAdd%Policy_sigma_readout/MatMul:product:03Policy_sigma_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Policy_sigma_readout/BiasAdda
AbsAbs%Policy_sigma_readout/BiasAdd:output:0*
T0*
_output_shapes

:2
AbsO
	Squeeze_1SqueezeAbs:y:0*
T0*
_output_shapes
:2
	Squeeze_1·
#Value_readout/MatMul/ReadVariableOpReadVariableOp,value_readout_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#Value_readout/MatMul/ReadVariableOp®
Value_readout/MatMulMatMul Value_layer_2/Relu:activations:0+Value_readout/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Value_readout/MatMul¶
$Value_readout/BiasAdd/ReadVariableOpReadVariableOp-value_readout_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$Value_readout/BiasAdd/ReadVariableOp°
Value_readout/BiasAddBiasAddValue_readout/MatMul:product:0,Value_readout/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Value_readout/BiasAddb
	Squeeze_2SqueezeValue_readout/BiasAdd:output:0*
T0*
_output_shapes
: 2
	Squeeze_2
IdentityIdentitySqueeze:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity

Identity_1IdentitySqueeze_1:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
:2

Identity_1

Identity_2IdentitySqueeze_2:output:0#^Policy_mu_0/BiasAdd/ReadVariableOp"^Policy_mu_0/MatMul/ReadVariableOp#^Policy_mu_1/BiasAdd/ReadVariableOp"^Policy_mu_1/MatMul/ReadVariableOp#^Policy_mu_2/BiasAdd/ReadVariableOp"^Policy_mu_2/MatMul/ReadVariableOp)^Policy_mu_readout/BiasAdd/ReadVariableOp(^Policy_mu_readout/MatMul/ReadVariableOp&^Policy_sigma_0/BiasAdd/ReadVariableOp%^Policy_sigma_0/MatMul/ReadVariableOp&^Policy_sigma_1/BiasAdd/ReadVariableOp%^Policy_sigma_1/MatMul/ReadVariableOp&^Policy_sigma_2/BiasAdd/ReadVariableOp%^Policy_sigma_2/MatMul/ReadVariableOp,^Policy_sigma_readout/BiasAdd/ReadVariableOp+^Policy_sigma_readout/MatMul/ReadVariableOp%^Value_layer_0/BiasAdd/ReadVariableOp$^Value_layer_0/MatMul/ReadVariableOp%^Value_layer_1/BiasAdd/ReadVariableOp$^Value_layer_1/MatMul/ReadVariableOp%^Value_layer_2/BiasAdd/ReadVariableOp$^Value_layer_2/MatMul/ReadVariableOp%^Value_readout/BiasAdd/ReadVariableOp$^Value_readout/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*}
_input_shapesl
j:::::::::::::::::::::::::2H
"Policy_mu_0/BiasAdd/ReadVariableOp"Policy_mu_0/BiasAdd/ReadVariableOp2F
!Policy_mu_0/MatMul/ReadVariableOp!Policy_mu_0/MatMul/ReadVariableOp2H
"Policy_mu_1/BiasAdd/ReadVariableOp"Policy_mu_1/BiasAdd/ReadVariableOp2F
!Policy_mu_1/MatMul/ReadVariableOp!Policy_mu_1/MatMul/ReadVariableOp2H
"Policy_mu_2/BiasAdd/ReadVariableOp"Policy_mu_2/BiasAdd/ReadVariableOp2F
!Policy_mu_2/MatMul/ReadVariableOp!Policy_mu_2/MatMul/ReadVariableOp2T
(Policy_mu_readout/BiasAdd/ReadVariableOp(Policy_mu_readout/BiasAdd/ReadVariableOp2R
'Policy_mu_readout/MatMul/ReadVariableOp'Policy_mu_readout/MatMul/ReadVariableOp2N
%Policy_sigma_0/BiasAdd/ReadVariableOp%Policy_sigma_0/BiasAdd/ReadVariableOp2L
$Policy_sigma_0/MatMul/ReadVariableOp$Policy_sigma_0/MatMul/ReadVariableOp2N
%Policy_sigma_1/BiasAdd/ReadVariableOp%Policy_sigma_1/BiasAdd/ReadVariableOp2L
$Policy_sigma_1/MatMul/ReadVariableOp$Policy_sigma_1/MatMul/ReadVariableOp2N
%Policy_sigma_2/BiasAdd/ReadVariableOp%Policy_sigma_2/BiasAdd/ReadVariableOp2L
$Policy_sigma_2/MatMul/ReadVariableOp$Policy_sigma_2/MatMul/ReadVariableOp2Z
+Policy_sigma_readout/BiasAdd/ReadVariableOp+Policy_sigma_readout/BiasAdd/ReadVariableOp2X
*Policy_sigma_readout/MatMul/ReadVariableOp*Policy_sigma_readout/MatMul/ReadVariableOp2L
$Value_layer_0/BiasAdd/ReadVariableOp$Value_layer_0/BiasAdd/ReadVariableOp2J
#Value_layer_0/MatMul/ReadVariableOp#Value_layer_0/MatMul/ReadVariableOp2L
$Value_layer_1/BiasAdd/ReadVariableOp$Value_layer_1/BiasAdd/ReadVariableOp2J
#Value_layer_1/MatMul/ReadVariableOp#Value_layer_1/MatMul/ReadVariableOp2L
$Value_layer_2/BiasAdd/ReadVariableOp$Value_layer_2/BiasAdd/ReadVariableOp2J
#Value_layer_2/MatMul/ReadVariableOp#Value_layer_2/MatMul/ReadVariableOp2L
$Value_readout/BiasAdd/ReadVariableOp$Value_readout/BiasAdd/ReadVariableOp2J
#Value_readout/MatMul/ReadVariableOp#Value_readout/MatMul/ReadVariableOp:K G

_output_shapes

:
%
_user_specified_nameinput_state
í

1__inference_Policy_sigma_0_layer_call_fn_14718402

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_147177302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
è
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_14718275

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
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
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö	
å
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_14718433

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç

.__inference_Policy_mu_1_layer_call_fn_14718362

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_147176762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_layer_2_layer_call_fn_14718502

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_147178652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_readout_layer_call_fn_14718322

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_readout_layer_call_and_return_conditional_losses_147179462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó	
â
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_14718353

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö	
å
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_14717730

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
ä
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_14717811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö	
å
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_14717784

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó	
â
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_14717703

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó

4__inference_Policy_mu_readout_layer_call_fn_14718284

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
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
GPU 2J 8 *X
fSRQ
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_147178912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

0__inference_Value_layer_0_layer_call_fn_14718462

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_147178112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î7
Þ

!__inference__traced_save_14718599
file_prefix7
3savev2_policy_mu_readout_kernel_read_readvariableop5
1savev2_policy_mu_readout_bias_read_readvariableop:
6savev2_policy_sigma_readout_kernel_read_readvariableop8
4savev2_policy_sigma_readout_bias_read_readvariableop3
/savev2_value_readout_kernel_read_readvariableop1
-savev2_value_readout_bias_read_readvariableop1
-savev2_policy_mu_0_kernel_read_readvariableop/
+savev2_policy_mu_0_bias_read_readvariableop1
-savev2_policy_mu_1_kernel_read_readvariableop/
+savev2_policy_mu_1_bias_read_readvariableop1
-savev2_policy_mu_2_kernel_read_readvariableop/
+savev2_policy_mu_2_bias_read_readvariableop4
0savev2_policy_sigma_0_kernel_read_readvariableop2
.savev2_policy_sigma_0_bias_read_readvariableop4
0savev2_policy_sigma_1_kernel_read_readvariableop2
.savev2_policy_sigma_1_bias_read_readvariableop4
0savev2_policy_sigma_2_kernel_read_readvariableop2
.savev2_policy_sigma_2_bias_read_readvariableop3
/savev2_value_layer_0_kernel_read_readvariableop1
-savev2_value_layer_0_bias_read_readvariableop3
/savev2_value_layer_1_kernel_read_readvariableop1
-savev2_value_layer_1_bias_read_readvariableop3
/savev2_value_layer_2_kernel_read_readvariableop1
-savev2_value_layer_2_bias_read_readvariableop
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
ShardedFilenameÉ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û	
valueÑ	BÎ	B,readout_mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB*readout_mu/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_sigma/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_sigma/bias/.ATTRIBUTES/VARIABLE_VALUEB/readout_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB-readout_value/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesº
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesâ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_policy_mu_readout_kernel_read_readvariableop1savev2_policy_mu_readout_bias_read_readvariableop6savev2_policy_sigma_readout_kernel_read_readvariableop4savev2_policy_sigma_readout_bias_read_readvariableop/savev2_value_readout_kernel_read_readvariableop-savev2_value_readout_bias_read_readvariableop-savev2_policy_mu_0_kernel_read_readvariableop+savev2_policy_mu_0_bias_read_readvariableop-savev2_policy_mu_1_kernel_read_readvariableop+savev2_policy_mu_1_bias_read_readvariableop-savev2_policy_mu_2_kernel_read_readvariableop+savev2_policy_mu_2_bias_read_readvariableop0savev2_policy_sigma_0_kernel_read_readvariableop.savev2_policy_sigma_0_bias_read_readvariableop0savev2_policy_sigma_1_kernel_read_readvariableop.savev2_policy_sigma_1_bias_read_readvariableop0savev2_policy_sigma_2_kernel_read_readvariableop.savev2_policy_sigma_2_bias_read_readvariableop/savev2_value_layer_0_kernel_read_readvariableop-savev2_value_layer_0_bias_read_readvariableop/savev2_value_layer_1_kernel_read_readvariableop-savev2_value_layer_1_bias_read_readvariableop/savev2_value_layer_2_kernel_read_readvariableop-savev2_value_layer_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
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

identity_1Identity_1:output:0*Ù
_input_shapesÇ
Ä: : :: :: :: : :  : :  : : : :  : :  : : : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :

_output_shapes
: 
	
ä
K__inference_Value_readout_layer_call_and_return_conditional_losses_14718313

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*÷
serving_defaultã
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ'
mu!
StatefulPartitionedCall:0*
sigma!
StatefulPartitionedCall:13
value_estimate!
StatefulPartitionedCall:2tensorflow/serving/predict:µ

mu_layer

readout_mu
sigma_layer
readout_sigma
value_layer
readout_value
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
__call__
_default_save_signature
+ &call_and_return_all_conditional_losses
	¡call"ï
_tf_keras_modelÕ{"class_name": "A2C", "name": "a2c_120", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "A2C"}}
5
0
1
2"
trackable_list_wrapper


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"Ý
_tf_keras_layerÃ{"class_name": "Dense", "name": "Policy_mu_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_readout", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
5
0
1
2"
trackable_list_wrapper


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "Dense", "name": "Policy_sigma_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_readout", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
5
0
1
 2"
trackable_list_wrapper
ü

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "Value_readout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_readout", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
Ö
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23"
trackable_list_wrapper
Ö
'0
(1
)2
*3
+4
,5
6
7
-8
.9
/10
011
112
213
14
15
316
417
518
619
720
821
!22
"23"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
9non_trainable_variables
trainable_variables
:metrics

;layers
	variables
<layer_metrics
=layer_regularization_losses
	regularization_losses
__call__
_default_save_signature
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
-
¨serving_default"
signature_map
õ

'kernel
(bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "Policy_mu_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 8]}}
÷

)kernel
*bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "Policy_mu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
÷

+kernel
,bias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
­__call__
+®&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "Policy_mu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_mu_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
*:( 2Policy_mu_readout/kernel
$:"2Policy_mu_readout/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Jnon_trainable_variables
Klayer_regularization_losses
trainable_variables
Lmetrics
	variables
Mlayer_metrics

Nlayers
regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
û

-kernel
.bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Policy_sigma_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 8]}}
ý

/kernel
0bias
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Policy_sigma_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
ý

1kernel
2bias
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
³__call__
+´&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "Policy_sigma_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Policy_sigma_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
-:+ 2Policy_sigma_readout/kernel
':%2Policy_sigma_readout/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
[non_trainable_variables
\layer_regularization_losses
trainable_variables
]metrics
	variables
^layer_metrics

_layers
regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
ù

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "Value_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_0", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 8]}}
û

5kernel
6bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Value_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
û

7kernel
8bias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"class_name": "Dense", "name": "Value_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Value_layer_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 32]}}
&:$ 2Value_readout/kernel
 :2Value_readout/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
lnon_trainable_variables
mlayer_regularization_losses
#trainable_variables
nmetrics
$	variables
olayer_metrics

players
%regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
$:" 2Policy_mu_0/kernel
: 2Policy_mu_0/bias
$:"  2Policy_mu_1/kernel
: 2Policy_mu_1/bias
$:"  2Policy_mu_2/kernel
: 2Policy_mu_2/bias
':% 2Policy_sigma_0/kernel
!: 2Policy_sigma_0/bias
':%  2Policy_sigma_1/kernel
!: 2Policy_sigma_1/bias
':%  2Policy_sigma_2/kernel
!: 2Policy_sigma_2/bias
&:$ 2Value_layer_0/kernel
 : 2Value_layer_0/bias
&:$  2Value_layer_1/kernel
 : 2Value_layer_1/bias
&:$  2Value_layer_2/kernel
 : 2Value_layer_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
 10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
qnon_trainable_variables
rlayer_regularization_losses
>trainable_variables
smetrics
?	variables
tlayer_metrics

ulayers
@regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
vnon_trainable_variables
wlayer_regularization_losses
Btrainable_variables
xmetrics
C	variables
ylayer_metrics

zlayers
Dregularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
{non_trainable_variables
|layer_regularization_losses
Ftrainable_variables
}metrics
G	variables
~layer_metrics

layers
Hregularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
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
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
Otrainable_variables
metrics
P	variables
layer_metrics
layers
Qregularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
Strainable_variables
metrics
T	variables
layer_metrics
layers
Uregularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
Wtrainable_variables
metrics
X	variables
layer_metrics
layers
Yregularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
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
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
`trainable_variables
metrics
a	variables
layer_metrics
layers
bregularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
dtrainable_variables
metrics
e	variables
layer_metrics
layers
fregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
htrainable_variables
metrics
i	variables
layer_metrics
layers
jregularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
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
ý2ú
*__inference_a2c_120_layer_call_fn_14718024Ë
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
input_1ÿÿÿÿÿÿÿÿÿ
á2Þ
#__inference__wrapped_model_14717634¶
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
input_1ÿÿÿÿÿÿÿÿÿ
2
E__inference_a2c_120_layer_call_and_return_conditional_losses_14717966Ë
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
input_1ÿÿÿÿÿÿÿÿÿ
ã2à
__inference_call_14718174
__inference_call_14718265§
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
Þ2Û
4__inference_Policy_mu_readout_layer_call_fn_14718284¢
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
ù2ö
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_14718275¢
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
á2Þ
7__inference_Policy_sigma_readout_layer_call_fn_14718303¢
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
ü2ù
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_14718294¢
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
Ú2×
0__inference_Value_readout_layer_call_fn_14718322¢
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
õ2ò
K__inference_Value_readout_layer_call_and_return_conditional_losses_14718313¢
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
ÍBÊ
&__inference_signature_wrapper_14718083input_1"
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
Ø2Õ
.__inference_Policy_mu_0_layer_call_fn_14718342¢
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
ó2ð
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_14718333¢
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
Ø2Õ
.__inference_Policy_mu_1_layer_call_fn_14718362¢
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
ó2ð
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_14718353¢
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
Ø2Õ
.__inference_Policy_mu_2_layer_call_fn_14718382¢
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
ó2ð
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_14718373¢
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
Û2Ø
1__inference_Policy_sigma_0_layer_call_fn_14718402¢
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
ö2ó
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_14718393¢
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
Û2Ø
1__inference_Policy_sigma_1_layer_call_fn_14718422¢
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
ö2ó
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_14718413¢
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
Û2Ø
1__inference_Policy_sigma_2_layer_call_fn_14718442¢
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
ö2ó
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_14718433¢
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
Ú2×
0__inference_Value_layer_0_layer_call_fn_14718462¢
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
õ2ò
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_14718453¢
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
Ú2×
0__inference_Value_layer_1_layer_call_fn_14718482¢
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
õ2ò
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_14718473¢
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
Ú2×
0__inference_Value_layer_2_layer_call_fn_14718502¢
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
õ2ò
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_14718493¢
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
 ©
I__inference_Policy_mu_0_layer_call_and_return_conditional_losses_14718333\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_0_layer_call_fn_14718342O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Policy_mu_1_layer_call_and_return_conditional_losses_14718353\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_1_layer_call_fn_14718362O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Policy_mu_2_layer_call_and_return_conditional_losses_14718373\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Policy_mu_2_layer_call_fn_14718382O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¯
O__inference_Policy_mu_readout_layer_call_and_return_conditional_losses_14718275\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_Policy_mu_readout_layer_call_fn_14718284O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¬
L__inference_Policy_sigma_0_layer_call_and_return_conditional_losses_14718393\-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_0_layer_call_fn_14718402O-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ¬
L__inference_Policy_sigma_1_layer_call_and_return_conditional_losses_14718413\/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_1_layer_call_fn_14718422O/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¬
L__inference_Policy_sigma_2_layer_call_and_return_conditional_losses_14718433\12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
1__inference_Policy_sigma_2_layer_call_fn_14718442O12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ²
R__inference_Policy_sigma_readout_layer_call_and_return_conditional_losses_14718294\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_Policy_sigma_readout_layer_call_fn_14718303O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
K__inference_Value_layer_0_layer_call_and_return_conditional_losses_14718453\34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_0_layer_call_fn_14718462O34/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_layer_1_layer_call_and_return_conditional_losses_14718473\56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_1_layer_call_fn_14718482O56/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_layer_2_layer_call_and_return_conditional_losses_14718493\78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
0__inference_Value_layer_2_layer_call_fn_14718502O78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ «
K__inference_Value_readout_layer_call_and_return_conditional_losses_14718313\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Value_readout_layer_call_fn_14718322O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÖ
#__inference__wrapped_model_14717634®'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "`ª]

mu

mu

sigma
sigma
+
value_estimate
value_estimate
E__inference_a2c_120_layer_call_and_return_conditional_losses_14717966¾'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "p¢m
fªc

mu
0/mu

sigma
0/sigma
-
value_estimate
0/value_estimate
 Ý
*__inference_a2c_120_layer_call_fn_14718024®'()*+,-./012345678!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "`ª]

mu

mu

sigma
sigma
+
value_estimate
value_estimateÉ
__inference_call_14718174«'()*+,-./012345678!"+¢(
!¢

input_state
ª "bª_

mu
mu

sigma
sigma
)
value_estimate
value_estimate Ð
__inference_call_14718265²'()*+,-./012345678!"4¢1
*¢'
%"
input_stateÿÿÿÿÿÿÿÿÿ
ª "`ª]

mu

mu

sigma
sigma
+
value_estimate
value_estimateä
&__inference_signature_wrapper_14718083¹'()*+,-./012345678!";¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"`ª]

mu

mu

sigma
sigma
+
value_estimate
value_estimate