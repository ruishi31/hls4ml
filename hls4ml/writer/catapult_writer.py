from __future__ import print_function
import tarfile
import yaml
from shutil import copyfile, copytree, rmtree
import numpy as np
import os
import re
import glob
from collections import OrderedDict
from hls4ml.writer.writers import Writer

from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.model.hls_layers import XnorPrecisionType
	
config_filename = 'hls4ml_config.yml'



class CatapultWriter(VivadoWriter):

    def print_array_to_cpp(self, var, odir, write_txt_file=True):
        #######################################
        ## Print weight array to C++
        #######################################

        h_file = open("{}/firmware/weights/{}.h".format(odir,var.name),"w")
        if write_txt_file:
            txt_file = open("{}/firmware/weights/{}.txt".format(odir,var.name),"w")

        #meta data
        h_file.write("//Numpy array shape {}\n".format(var.shape))
        h_file.write("//Min {:.12f}\n".format(np.min(var.min)))
        h_file.write("//Max {:.12f}\n".format(np.max(var.max)))
        h_file.write("//Number of zeros {}\n".format(var.nzeros))
        h_file.write("\n")

        h_file.write("#ifndef {}_H_\n".format(var.name.upper()))
        h_file.write("#define {}_H_\n".format(var.name.upper()))
        h_file.write("\n")

        if write_txt_file:
            h_file.write("#ifndef __SYNTHESIS__\n")
            h_file.write("static " + var.definition_cpp() + ";\n")
            h_file.write("#else\n")

        h_file.write("static " + var.definition_cpp() + " = {")

        #fill c++ array.
        #not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ", "
        h_file.write("};\n")
        if write_txt_file:
            h_file.write("#endif\n")
            txt_file.close()
        h_file.write("\n#endif\n")
        h_file.close()

    def write_project_dir(self, model):
        if not os.path.isdir("{}/firmware/weights".format(model.config.get_output_dir())):
            os.makedirs("{}/firmware/weights".format(model.config.get_output_dir()))

    @staticmethod
    def _make_array_pragma(variable):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """
        
        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '//#pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}' # It may be needed for catpult optimization in the future, so just keep them here for now
            else:
                template = '//#pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}' # It may be needed for catpult optimization in the future, so just keep them here for now

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            return '//#pragma HLS STREAM variable={name} depth={depth} dim={dim}'.format(name=variable.name, depth=depth, dim=0) # It may be needed for catpult optimization in the future, so just keep them here for now

    @staticmethod	
    def _make_stable_pragma(variable):	
        template = '//#pragma HLS STABLE variable={name}'# It may be needed for catpult optimization in the future, so just keep them here for now	
        return template.format(name=variable.name)

    def write_project_cpp(self, model):
        ###################
        ## myproject.cpp
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/catapult/firmware/myproject.cpp'),'r')
        fout = open('{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():
            #Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp() for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp() for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(i) for i in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + ',\n'
            elif '//hls-fpga-machine-learning insert weights ports' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + '{} {}[{}],\n'.format(w.type.name, w.name, w.nonzeros)
                        else:
                            newline += indent + '{} {}[{}],\n'.format(w.type.name, w.name, w.data_length)
                newline = newline[:-2] + '\n'
            elif '//hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.nonzeros, w.name, w.name)
                        elif w.__class__.__name__ == 'ExponentWeightVariable':	
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.data_length, w.name, w.name)
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.data_length, w.name, w.name)

            #Add input/output type
            elif '//hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.cppname for i in model_inputs]
                all_outputs = [o.cppname for o in model_outputs]

                if model.config.get_config_value("IOType") == "io_parallel":
                    for i in model_inputs: newline += indent + self._make_array_pragma(i) + '\n'
                    for o in model_outputs: newline += indent + self._make_array_pragma(o) + '\n'
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays (16.03.2020)
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '//#pragma HLS INTERFACE ap_vld port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                    if model.config.model_strategy == 'Resource':
                        newline += indent + '//#pragma HLS DATAFLOW \n'
                    else:
                        newline += indent + '//#pragma HLS PIPELINE \n'
                if model.config.get_config_value("IOType") == "io_serial":
                    newline += indent + '//#pragma HLS INTERFACE axis port={},{} \n'.format(','.join(all_inputs), ','.join(all_outputs))
                    newline += indent + '//#pragma HLS DATAFLOW \n'

                inval_str = '\n    '.join(['const_size_in_{} = {};'.format(i, inp.size_cpp()) for i, inp in enumerate(model_inputs, 1)])
                outval_str = '\n    '.join(['const_size_out_{} = {};'.format(i, out.size_cpp()) for i, out in enumerate(model_outputs, 1)])
                newline += '\n' + indent + inval_str
                newline += '\n' + indent + outval_str
                newline += '\n'

            elif '//hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                inputs = model.get_input_variables()
                outputs = model.get_output_variables()
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in inputs and var not in outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n'
                                if model.config.model_strategy == 'Resource':	
                                    newline += '    ' + self._make_stable_pragma(var) + '\n'
                    func = layer.function_cpp()
                    if func:
                        for line in func:
                            newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('Trace', False):
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(var.type.name, var.name, layer.name, var.size_cpp())
                            newline += '#endif\n'
                        newline += '\n'

            #Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        #######################
        ## myproject.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/catapult/firmware/myproject.h'),'r')
        fout = open('{}/firmware/{}.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp() for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp() for o in model_outputs])
                insize_str = ', '.join(['unsigned short &const_size_in_{}'.format(i) for i in range(1, len(model_inputs) + 1)])
                outsize_str = ', '.join(['unsigned short &const_size_out_{}'.format(o) for o in range(1, len(model_outputs) + 1)])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + ',\n'
                newline += indent + insize_str + ',\n'
                newline += indent + outsize_str + ',\n'
            elif '//hls-fpga-machine-learning insert weights ports' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + '{} {}[{}],\n'.format(w.type.name, w.name, w.nonzeros)
                        else:
                            newline += indent + '{} {}[{}],\n'.format(w.type.name, w.name, w.data_length)
                newline = newline[:-2] + '\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/catapult/firmware/defines.h'),'r')
        fout = open('{}/firmware/defines.h'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            #Insert numbers
            if '//hls-fpga-machine-learning insert numbers' in line:
                newline = line
                numbers = OrderedDict.fromkeys([layer.get_numbers_cpp() for layer in model.get_layers()])
                newline += ''.join(numbers)

            elif '//hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    all_precision.update(layer_precision)
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/catapult/firmware/parameters.h'),'r')
        fout = open('{}/firmware/parameters.h'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            if '//hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.include_list for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif '//hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += '#include "weights/{}.h"\n'.format(w.name)

            elif "//hls-fpga-machine-learning insert layer-config" in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.config_cpp()
                    if config:
                        newline += config + '\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, model.config.get_output_dir())
    
    def __make_dat_file(self, original_path, project_path): 
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter. 
        """

        #Take in data from current supported data files
        if original_path[-3:] == "npy":
            data = np.load(original_path)
        else:
            raise Exception("Unsupported input/output data files.")

        #Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + " ")
                f.write("\n")

        #Print out in dat file
        with open(project_path, "w" ) as f:
            print_data(f)

    def write_test_bench(self, model):
        ###################
        ## test bench
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists('{}/tb_data/'.format(model.config.get_output_dir())):
            os.mkdir('{}/tb_data/'.format(model.config.get_output_dir()))
        
        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')
        
        if input_data:
            if input_data[-3:] == "dat":
                copyfile(input_data, '{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(input_data,'{}/tb_data/tb_input_features.dat'.format(model.config.get_output_dir()))
        
        if output_predictions:
            if output_predictions[-3:] == "dat":
                copyfile(output_predictions, '{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))
            else:
                self.__make_dat_file(output_predictions,'{}/tb_data/tb_output_predictions.dat'.format(model.config.get_output_dir()))

        f = open(os.path.join(filedir,'../templates/catapult/myproject_test.cpp'),'r')
        fout = open('{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            #Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls-fpga-machine-learning weights headfiles' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + '#include "firmware/weights/{}.h"\n'.format(w.name)
                        else:
                            newline += indent + '#include "firmware/weights/{}.h"\n'.format(w.name)
            elif '//hls-fpga-machine-learning insert data' in line:
                newline = line
                newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
                newline += '      std::vector<float>::const_iterator in_end;\n'
                for inp in model.get_input_variables():
                    newline += '      ' + inp.definition_cpp() + ';\n'
                    newline += '      in_end = in_begin + ({});\n'.format(inp.size_cpp())
                    newline += '      std::copy(in_begin, in_end, {});\n'.format(inp.cppname)
                    newline += '      in_begin = in_end;\n'
                for out in model.get_output_variables():
                    # brace-init zeros the array out because we use std=c++0x
                    newline += '      ' + out.definition_cpp() + '{};\n'
                    # but we can still explicitly zero out if you want
                    newline += '      std::fill_n({}, {}, 0.);\n'.format(out.cppname, out.size())
            elif '//hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model.get_input_variables():
                    newline += '    ' + inp.definition_cpp() + ';\n'
                    newline += '    std::fill_n({}, {}, 0.);\n'.format(inp.cppname, inp.size_cpp())
                for out in model.get_output_variables():
                    newline += '    ' + out.definition_cpp() + '{};\n'
                    newline += '      std::fill_n({}, {}, 0.);\n'.format(out.cppname, out.size())
            elif '//hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                size_str = indent + 'unsigned short {},{};\n'
                input_size_vars = ','.join(['size_in{}'.format(i) for i in range(1, len(model.get_input_variables()) + 1)])
                output_size_vars = ','.join(['size_out{}'.format(o) for o in range(1, len(model.get_output_variables()) + 1)])
                newline += size_str.format(input_size_vars, output_size_vars)

                input_vars = ','.join([i.cppname for i in model.get_input_variables()])
                output_vars = ','.join([o.cppname for o in model.get_output_variables()])
                top_level = indent + 'CCS_DESIGN({})({},{},{},{}'.format(model.config.get_project_name(), input_vars, output_vars, input_size_vars, output_size_vars)
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        top_level += ',{}'.format(w.name)
                top_level += ');\n'
                newline += top_level
            elif '//hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  fout << {}[i] << " ";\n'.format(out.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'fout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
                newline = line
                for out in model.get_output_variables():
                    newline += indent + 'for(int i = 0; i < {}; i++) {{\n'.format(out.size_cpp())
                    newline += indent + '  std::cout << {}[i] << " ";\n'.format(out.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '//hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            newline += indent + 'nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.nonzeros, w.name, w.name)
                        else:
                            newline += indent + 'nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(w.type.name, w.data_length, w.name, w.name)
            elif '//save traces step1' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += indent + 'std::string {}_FILE_BIN_MEM = "tb_data/{}.mem";\n'.format(w.name.upper(), w.name)
                        newline += indent + 'std::ofstream fout_{}fbm({}_FILE_BIN_MEM);\n'.format(w.name, w.name.upper())
            elif '//save traces step2' in line:
                newline = line

                for inp in model.get_input_variables():
                    newline += indent + 'for(int i = size_in1-1; i >=0; i--) {\n'
                    newline += indent + '  print_fxd_as_bin<input_t>(fout_ifbm, {}[i]);\n'.format(inp.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'fout_ifbm << std::endl;\n'

                for out in model.get_output_variables():
                    newline += indent + 'for(int i = {}-1; i >=0; i--) {{\n'.format(out.size_cpp())
                    newline += indent + '  print_fxd_as_bin<result_t>(fout_ofbm, {}[i]);\n'.format(out.cppname)
                    newline += indent + '}\n'
                    newline += indent + 'fout_ofbm << std::endl;\n'

                newline += indent + 'static bool one_time = true;\n'
                newline += indent + 'if (one_time) {\n'
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += indent + 'unsigned int size_{} = {};\n'.format(w.name, w.data_length)
                        newline += indent + 'for(int i = size_{}-1; i >= 0; i--) {{\n'.format(w.name)
                        newline += indent + '  print_fxd_as_bin<{}>(fout_{}fbm, {}[i]);\n'.format(w.type.name, w.name, w.name)
                        newline += indent + '}\n'
                        newline += indent + 'fout_{}fbm << std::endl;\n'.format(w.name)
                newline += indent + 'one_time = false;\n'
                newline += indent + '}\n'
                            
            elif '//save traces step3' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += indent + 'fout_{}fbm.close();\n'.format(w.name)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    
    def write_build_script(self, model):
        ###################
        # build_prj.tcl
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir,'../templates/catapult/build_prj.tcl'),'r')
        fout = open('{}/build_prj.tcl'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():

            line = line.replace('myproject',model.config.get_project_name())

            if '#set_working_dir' in line:
                line = 'set_working_dir {}\n'.format(os.getcwd()+ '/' + model.config.get_output_dir())
            elif 'add ../OutputDir/firmware/ProjectName.cpp' in line:
                line = line.replace('add ../OutputDir/firmware/ProjectName.cpp','add ../{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()))
            elif '../OutputDir/ProjectName_test.cpp' in line:
                line = line.replace('../OutputDir/ProjectName_test.cpp','../{}/{}_test.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()))
            elif 'OutputDir' in line:
                line = line.replace('OutputDir',model.config.get_output_dir())
            elif 'ProjectName' in line:
                line = line.replace('ProjectName',model.config.get_project_name())
            elif '-CLOCK_PERIOD 25.0 \\\n' in line:
                line = line.replace('-CLOCK_PERIOD 25.0 \\\n', '-CLOCK_PERIOD {} \\\n'.format(model.config.get_config_value('ClockPeriod')))

            fout.write(line)
        f.close()
        fout.close()

        ###################
        # build_lib.sh
        ###################

        f = open(os.path.join(filedir,'../templates/catapult/build_lib.sh'),'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():
            line = line.replace('myproject', model.config.get_project_name())

            fout.write(line)
        f.close()
        fout.close()

    def write_nnet_utils(self, model):
        ###################
        ## nnet_utils
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/catapult/nnet_utils/')
        dstpath = '{}/firmware/nnet_utils/'.format(model.config.get_output_dir())

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        ###################
        ## ap_types
        ###################

        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir,'../templates/catapult/ap_types/')
        dstpath = '{}/firmware/ap_types/'.format(model.config.get_output_dir())

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)
        
    def write_yml(self, model):	
        ###################	
        # YAML config file	
        ###################	
        def keras_model_representer(dumper, keras_model):	
            model_path = model.config.get_output_dir() + '/keras_model.h5'	
            keras_model.save(model_path)	
            return dumper.represent_scalar(u'!keras_model', model_path)	
        try:	
            from tensorflow.keras import Model as KerasModel	
            yaml.add_multi_representer(KerasModel, keras_model_representer)	
        except:	
            pass	
        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:	
            yaml.dump(model.config.config, file)	


    def write_tar(self, model):
        ###################
        # Tarball output
        ###################

        with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
            archive.add(model.config.get_output_dir(), recursive=True)

    def write_hls(self, model):
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Writing HLS project - the catapult one')
        print('Done')

