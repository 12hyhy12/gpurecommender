import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from memory_profiler import profile
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
import sys
import timeit
import os
import psutil

profile_count=0
profile_point=21

class op_graph:
    def __init__(self):
        self.dictionary = {}
        self.var_nodes = []
        self.call_nodes = []

    #type "call" means result of function
    #type "var" means variable
    def insert_op(self, current_op_node):
        if self.find_if_exist(current_op_node) == None:
            self.dictionary[current_op_node.id] = current_op_node
            if current_op_node.type == "call":
                self.call_nodes.append(current_op_node)
            if current_op_node.type == "var":
                self.var_nodes.append(current_op_node)
    
    def find_starting_ops(self):
        result = []
        for temp_op in self.call_nodes:
            if_started = True 
            for key in temp_op.prior.keys():
                if temp_op.prior[key][1].type == "call":
                    if_started = False
                    break
            if if_started == True:
                result.append(temp_op)
        return result

    def find_if_exist(self, current_op_node):
        for key in self.dictionary.keys():
            if key == current_op_node.id:
                return self.dictionary[key]
        return None

    def find_if_var(self, var_op):
        for temp in self.var_nodes:
            if temp.op_instance == var_op:
                # print("find same params")
                return temp
        return None

    def print_graph(self, ir_params, x):
        # verify is there any replication.
        start_call_nodes = self.find_starting_ops()
        print("Total start call node with ops:", len(start_call_nodes))
        for start_op in start_call_nodes:
            start_op.print_self()
    
    def check_op_ready(self, temp_op):
        for key in temp_op.prior.keys():
            if temp_op.prior[key][1].type == "call":
                if "fw_value" not in temp_op.prior[key][1].performance_data.keys():
                    return False
                else:
                    # print("intermeidiate_arg",temp_op.prior[key][1].performance_data["fw_value"].shape, temp_op.prior[key][1].performance_data["fw_value"].dtype)
                    return True
        return True

    #bfs
    def traverse_and_calculate_per_op(self, ir_params, x, fw=True, bw=False):
        global profile_count
        available_op_queue = self.find_starting_ops()
        profile_count = 0
        N = 100
        while len(available_op_queue) > 0 :
            temp_op = available_op_queue.pop(0)
            for key in temp_op.next.keys():
                print("temp_op_son:" + str(temp_op.next[key][1].id))
            if fw:
                profile_forward_relay_operator(temp_op, ir_params, x)
            if bw:
                profile_backward_relay_operator(temp_op, ir_params, x)
            for key in temp_op.next.keys():
                if self.check_op_ready(temp_op.next[key][1]):
                    available_op_queue.append(temp_op.next[key][1])
            profile_count +=1

class op_node:
    # args_index:{id,}
    def __init__(self, type, current_index, op, attrs = None):
        self.type = type
        if self.type == "var":
            self.name = op.name_hint
        if self.type == "call":
            self.name = op.op.name
            self.attrs = attrs
        self.id = self.name + "-" + str(current_index)
        self.op_instance = op
        self.next = {}
        self.prior = {}
        self.performance_data = {}
    
    def set_next(self, next_op_node, next_op_index, args_index):
        next_op_id = next_op_node.name + "-" + str(next_op_index)
        self.next[next_op_id] = (args_index, next_op_node)
        next_op_node.prior[self.id] = (args_index, self)
    
    def print_self(self):
        print(self.id, self.attrs, self.next, self.prior)

op_index = 0
computation_graph = op_graph()

def construct_op_graph(ir_module):
    global op_index, computation_graph
    entrance_tuple = ir_module.functions.items()[0]
    global_var = entrance_tuple[0]
    main_function = entrance_tuple[1]

    # add all params (without intermeidiate_args)
    for each_param in main_function.params:
        temp_op_node = op_node("var", op_index, each_param, attrs = None)
        computation_graph.insert_op(temp_op_node)
        op_index+=1
    recursive_traverse_op(main_function.body.attrs, main_function.body.args, temp_op=main_function.body)
    print('main_function.body.attrs:' + str(main_function.body.attrs))
    print('main_function.body.args:' + str(main_function.body.args))
    #print(computation_graph.call_nodes)

def profile_memory(ir_params, x):
    computation_graph.traverse_and_calculate_per_op( ir_params, x, bw = True)


def recursive_traverse_op(attrs, args, temp_op=None):
    global op_index, computation_graph
    args_index = 0
    next_op_node = op_node("call", op_index, temp_op, attrs = attrs)
    op_index+=1
    for each_arg in args:
        if isinstance(each_arg, tvm.relay.expr.Call):
            current_node_op = recursive_traverse_op(each_arg.attrs, each_arg.args, temp_op = each_arg)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            args_index+=1
        if isinstance(each_arg,tvm.relay.expr.Var):
            current_node_op = computation_graph.find_if_var(each_arg)
            if current_node_op == None:
                current_node_op = op_node("var", op_index, each_arg, attrs = None)
            current_node_op.set_next(next_op_node, op_index-1, args_index)
            computation_graph.insert_op(current_node_op)
            args_index+=1
    computation_graph.insert_op(next_op_node)
    return next_op_node

def get_op_args(ready_op_node, dtype, params, x):
    intermeidiate_args = []
    args_index = 0
    pick_one = True
    max_index = 0
    for key in ready_op_node.prior.keys():
        if max_index < ready_op_node.prior[key][0]:
            max_index = ready_op_node.prior[key][0]
    while args_index <= max_index:
        for key in ready_op_node.prior.keys():
            if ready_op_node.prior[key][0] == args_index:
                if ready_op_node.prior[key][1].type == "call":
                    # need to append intermeidiate_args:
                    intermeidiate_args.append(ready_op_node.prior[key][1].performance_data["fw_value"])
                if ready_op_node.prior[key][1].type == "var":
                    # need to append params:
                    if ready_op_node.prior[key][1].name == '1' or ready_op_node.prior[key][1].name == 'input.1':
                        # find x:
                        # print("find x")
                        intermeidiate_args.append(x.astype(dtype))
                    else:
                        # may be this is not necessary since its keywork arguments.
                        # print("find keyword arguments")
                        # intermeidiate_args.append(params[ready_op_node.prior[key][1].name])
                        pass
                args_index+=1
    return intermeidiate_args

def find_nd_array_args(ready_op_node, args_index):
    for id_key in ready_op_node.prior.keys():
        temp_index = ready_op_node.prior[id_key][0]
        temp_prior_node = ready_op_node.prior[id_key][1]
        if temp_index == args_index:
            return temp_prior_node.performance_data["fw_value"].shape, temp_prior_node.performance_data["fw_value"].dtype
    print("cannot find ", ready_op_node.id, "'s intermeidiate_arg in index :", args_index)
    return None, None

def generate_intermediate_symbolic_args(ready_op_node):
    args_index = 0
    new_args = []
    for tvm_arg in ready_op_node.op_instance.args:
        if isinstance(tvm_arg, tvm.relay.expr.Call):
            s, d = find_nd_array_args(ready_op_node, args_index)
            temp_arg = tvm.relay.var(str(args_index), shape=s, dtype=d)
            new_args.append(temp_arg)
        if isinstance(tvm_arg, tvm.relay.expr.Var):
            new_args.append(tvm_arg)
        args_index+=1
    return new_args

def profile_forward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    global profile_count, profile_point
    if ready_op_node.type == "var":
        return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    print('new_args:' + str(new_args))
    temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs)
    #print('temp_body:' + str(temp_body))
    call_function = tvm.relay.Function(new_args, temp_body)
    call_functions = {"GlobalVar": None, "main": call_function}
    call_ir_module = tvm.ir.IRModule(functions=call_functions)
    #print(call_ir_module)
    with tvm.transform.PassContext(opt_level=1):
        call_interpreter = relay.build_module.create_executor("graph", call_ir_module, tvm.cuda(0), "cuda")
    call_intput_args = get_op_args(ready_op_node, dtype, ir_params, x)
    call_params = ir_params
    # np_running_time = timeit.timeit(
    #     setup="import timeit\n"
    #     "@profile\n"
    #     "def op_forwardASAA():\n"
    #     '  tvm_output = call_interpreter.evaluate()(*call_intput_args, **call_params)\n'
    #     '  print(\'goodaaaaaaa\')\n',
    #     stmt="op_forwardASAA()",
    #     number=1,
    #     globals=globals()
    # )
    print(ready_op_node.id)
    @profile
    def op_forward_profile():
        res = call_interpreter.evaluate()(*call_intput_args, **ir_params)
        return res
    
    def op_forward():
        res = call_interpreter.evaluate()(*call_intput_args, **ir_params)
        return res
    
    if profile_count == profile_point:
        ready_op_node.performance_data["fw_value"] = op_forward_profile()
    else:
        ready_op_node.performance_data["fw_value"] = op_forward()
    return 

def profile_backward_relay_operator(ready_op_node, ir_params, x, dtype="float32"):
    global profile_count, profile_point
    if ready_op_node.type == "var":
        return
    new_args = generate_intermediate_symbolic_args(ready_op_node)
    temp_body = tvm.relay.Call(ready_op_node.op_instance.op, new_args, attrs=ready_op_node.op_instance.attrs)
    call_function = tvm.relay.Function(new_args, temp_body)
    call_function = run_infer_type(call_function)
    bwd_func = run_infer_type(gradient(call_function))
    # compile in a different way:
    call_interpreter = relay.create_executor(device = tvm.cuda(0), target = "cuda")
    # prepare to run
    call_intput_args = get_op_args(ready_op_node, dtype, ir_params, x)
    print(ready_op_node.id)
    @profile
    def op_backward_profile():
        res = call_interpreter.evaluate(bwd_func)(*call_intput_args, **ir_params)
        return res
    
    def op_backward():
        res = call_interpreter.evaluate(bwd_func)(*call_intput_args, **ir_params)
        return res
    
    if profile_count == profile_point:
        ready_op_node.performance_data["bw_value"] = op_backward_profile()
    else:
        ready_op_node.performance_data["bw_value"] = op_backward()
    return 
