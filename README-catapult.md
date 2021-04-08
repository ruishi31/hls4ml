# hls4ml-catapult-backend

# Important

The catapult backend is developed based on the hls4ml(https://github.com/fastmachinelearning/hls4ml) updated on Apr 8th.


## These files were modified 

(Thanks Sioni, Hamza and Vladimir provided the method to add a new backend):

hls4ml/model/hls_layers.py

hls4ml/templates/__init__.py

hls4ml/writer/__init__.py


## These files were added:

hls4ml/templates/catapult/

hls4ml/templates/catapult_template.py

hls4ml/writer/catapult_writer.py

catapult-test/


# Summary:

For catapult backend, only the project under catapult-test/ is supported for the time being, and unoptimized. To support more project, the function called in 'hls4ml/templates/catapult/nnet_utils' should be modified based on specific needs (eg. Find the corresponding function in Catapult Document to substitude the Vivado one with the corresponding precision and settings)


## Functional correctness verification of a generated catapult project 

(Thanks Giuseppe provided this method of verifying the correctness, also provided the changes from ap_ to ac_ and etc.)

Type command:

conda activate hls4ml-tutorial

pip install -e .


### For Vivado:

cd catapult-test/test2

hls4ml convert -c keras-config.yml

hls4ml build -p keras_econ_4x4_d16 -c -s

hls4ml report -p keras_econ_4x4_d16

#### For the example project

cd catapult-test/test2

hls4ml convert -c keras-config.yml

cd keras_econ_4x4_d16

vim econ_4x4_d16_test.cpp (Please change '#define CHECKPOINT 5000' into '#define CHECKPOINT 1'. This is for functional correctness verification.)

cd ..

hls4ml build -p keras_econ_4x4_d16 -c (The terminal output is the output1.)

### For Catapult:

cd catapult-test/test1

hls4ml convert -c keras-config.yml

cd keras_econ_4x4_d16

catapult -product ultra -file build_prj.tcl (Commands may change based on different machines.)

#### For the example project(unoptimized)

cd catapult-test/test1

hls4ml convert -c keras-config.yml

cd keras_econ_4x4_d16

catapult -product ultra -file build_prj.tcl (Commands may change based on different machines.)

close the GUI 

cd tb_data

cat catapult_csim_results.log (The terminal output is the output2.)

Compare the last line of output1 and output 2: Quantized predictions. If the results are generally same, then the generated catapult project is assumed to be functional correct for now.

#### Example last-line output

Quantized predictions

Vivado: 1.47949 0.973633 1.2666 0.824219 0.972656 1.05371 1.9043 1.70508 0.875977 0.525391 1.21484 1.06641 0.731445 1.39844 0.772461 1.68945

Catapult: 1.4794921875 .9736328125 1.2666015625 .82421875 .97265625 1.0537109375 1.904296875 1.705078125 .8759765625 .525390625 1.21484375 1.06640625 .7314453125 1.3984375 .7724609375 1.689453125

