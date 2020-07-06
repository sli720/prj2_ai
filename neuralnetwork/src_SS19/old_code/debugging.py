# some code snippets which I used for debugging

"""
---------------------------------------------------------------
uint8
min = 0
max = 255
---------------------------------------------------------------
int8
min = - 128
max = 127
---------------------------------------------------------------
uint16
min = 0
max = 65535
---------------------------------------------------------------
int16
min = - 32768
max = 32767
---------------------------------------------------------------
uint32
min = 0
max = 4294967295
---------------------------------------------------------------
int32
min = - 2147483648
max = 2147483647
---------------------------------------------------------------
uint64
min = 0
max = 18446744073709551615
---------------------------------------------------------------
int64
min = - 9223372036854775808
max = 9223372036854775807
---------------------------------------------------------------
float16
min =        - 6.55040 e+04
max =        6.55040 e+04
---------------------------------------------------------------
float32
min =        - 3.4028235 e+38
max =        3.4028235 e+38
---------------------------------------------------------------
float64
min =        - 1.7976931348623157 e+308
max =        1.7976931348623157 e+308
---------------------------------------------------------------
"""
# For debugging
def print_datatype_ranges():
    import numpy as np
    int_types = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"]
    for it in int_types:
        print(np.iinfo(it))

    float_types = ["float16", "float32", "float64"]
    for ft in float_types:
        print(np.finfo(ft))
		
		
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
		





filename_friday_02_03_2018 = "Friday-02-03-2018.csv"
file_friday_02_03_2018_skip_rows = []


filename_friday_16_02_2018 = "Friday-16-02-2018.csv"
file_friday_16_02_2018_skip_rows = [1000000]


filename_friday_23_02_2018 = "Friday-23-02-2018.csv"
file_friday_23_02_2018_skip_rows = []


filename_thuesday_20_02_2018 = "Thuesday-20-02-2018.csv"
file_thuesday_20_02_2018_skip_rows = []


filename_thursday_01_03_2018 = "Thursday-01-03-2018.csv"
file_thursday_01_03_2018_skip_rows = [414,19762,19907,39020,60810,76529,81060,85449,89954,91405,92658,95061,331113,331114,331115,331116,331117,331118,331119,331120,331121,331122,331123,331124,331125]


filename_thursday_15_02_2018 = "Thursday-15-02-2018.csv"
file_thursday_15_02_2018_skip_rows = []


filename_thursday_22_02_2018 = "Thursday-22-02-2018.csv"
file_thursday_22_02_2018_skip_rows = []


filename_wednesday_14_02_2018 = "Wednesday-14-02-2018.csv"
file_wednesday_14_02_2018_skip_rows = []


filename_wednesday_21_02_2018 = "Wednesday-21-02-2018.csv"
file_wednesday_21_02_2018_skip_rows = []


filename_wednesday_28_02_2018 = "Wednesday-28-02-2018.csv"
file_wednesday_28_02_2018_skip_rows = [21839,43118,63292,84014,107720,132410,154206,160207,202681,228584,247718,271677,296995,322939,344163,349510,355080,360661,366040,367414,368614,371160,377705,399544,420823,440997,461719,485425,510115,534074,559392,585336,606560]







import ast
import numbers              
def is_numeric(obj):
    if isinstance(obj, numbers.Number):
        return True
    elif isinstance(obj, str):
        nodes = list(ast.walk(ast.parse(obj)))[1:]
        if not isinstance(nodes[0], ast.Expr):
            return False
        if not isinstance(nodes[-1], ast.Num):
            return False
        nodes = nodes[1:-1]
        for i in range(len(nodes)):
            #if used + or - in digit :
            if i % 2 == 0:
                if not isinstance(nodes[i], ast.UnaryOp):
                    return False
            else:
                if not isinstance(nodes[i], (ast.USub, ast.UAdd)):
                    return False
        return True
    else:
        return False
		
		
		


# Replace string based output labels ("benign", "infiltration", ...) with numbers for the classes (e.g. 0, 1, ...)
#encoder = LabelEncoder()
#encoder.fit(data_train_y)
#data_train_y = encoder.transform(data_train_y)
#data_test_y = encoder.transform(data_test_y)


"""
selected_features = ['dst_host_same_srv_rate',
 'dst_host_same_src_port_rate',
 'same_srv_rate',
 'dst_bytes',
 'diff_srv_rate',
 'dst_host_serror_rate',
 'duration',
 'rerror_rate',
 'logged_in',
 'dst_host_rerror_rate',
 'hot',
 'num_failed_logins',
 'srv_serror_rate',
 'dst_host_srv_serror_rate',
 'flag',
 'protocol_type',
 'dst_host_srv_rerror_rate',
 'dst_host_diff_srv_rate',
 'dst_host_srv_diff_host_rate',
 'serror_rate',
 'src_bytes',
 'service',
 'dst_host_count',
 'dst_host_srv_count',
 'count']

 """
 