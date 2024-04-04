#!/bin/bash

# Extract a value using yq
get_yaml_value() {
    local path=$1
    cat $config_file | yq .$path
}

config_file="testConfig.yml"

# S=$(get_yaml_value "S")
# E=$(get_yaml_value "E")
# P=$(get_yaml_value "P")
# H=$(get_yaml_value "H")

cores=$(get_yaml_value "cores")
board=$(get_yaml_value "board")
board=${board:1:-1}

listS=$(get_yaml_value "S[]")
listE=$(get_yaml_value "E[]")
listP=$(get_yaml_value "P[]")
listH=$(get_yaml_value "H[]")

testList=$(get_yaml_value "testToRun[]")

for S in $listS; do
    for E in $listE; do
        for P in $listP; do
            # P=$E
            for H in $listH; do
                idx=0
                for test in $testList; do
                    kernel_name=$(get_yaml_value $test.kernelName)
                    app_folder=$(get_yaml_value $test.appFolder)
                    platform=$(get_yaml_value $test.platform)
                    kernel_name=${kernel_name:1:-1}
                    app_folder=${app_folder:1:-1}
                    platform=${platform:1:-1}
                    test_name=${test:1:-1}

                    rm -rf $app_folder
                    mkdir -p $app_folder

                    if [ "$platform" == "ARM_QEMU" ]; then
                        echo "ARM config"
                    
                    else
                        mkdir -p $app_folder/inc
                        mkdir -p $app_folder/src
                        touch $app_folder/gvsoc.log

                        echo "Test $test:"
                        echo -e "\t S: $S"
                        echo -e "\t E: $E"
                        echo -e "\t P: $P"
                        echo -e "\t H: $H"
                        echo -e "\t kernel_name: $kernel_name"
                        echo -e "\t app_folder: $app_folder"

                        # Generate and save golden I/O and create the template
                        python generateIoAndTemplate.py --MHSA_params $S $E $P $H --kernel_name $kernel_name --app_folder $app_folder --board $board --test_idx $idx $1 $2
                        idx=$((idx + 1))

                        # Run the test and dump outputs
                        echo "Running the test..."
                        echo "make clean -C $app_folder"
                        make clean -C $app_folder >/dev/null
                        if [ ul == $platform ]; then # Platfrom is undefined, fallback to GVSoC
                            echo "make all -j -C $app_folder CORE=$cores platform=gvsoc > /dev/null"
                            make all -j -C $app_folder CORE=$cores platform=gvsoc > /dev/null
                            echo "make run -C $app_folder CORE=$cores platform=gvsoc > $app_folder/gvsoc.log"
                            make run -C $app_folder CORE=$cores platform=gvsoc > $app_folder/gvsoc.log
                        else
                            echo "make all -j -C $app_folder CORE=$cores platform=$platform > /dev/null"
                            make all -j -C $app_folder CORE=$cores platform=$platform > /dev/null
                            if [ "$platform" == "gvsoc" ]; then                            
                                make run -C $app_folder CORE=$cores platform=$platform > $app_folder/gvsoc.log
                                echo "Test ended, check the log file for the output"
                            else
                                echo "make run -C $app_folder CORE=$cores platform=$platform > $app_folder/gvsoc.log"
                                make run -C $app_folder CORE=$cores platform=$platform  > $app_folder/gvsoc.log
                            fi
                        fi
                        
                        if [ $test_name != "MHSA" ] && [ $test_name != "MHSAFWA" ] && [ $test_name != "MHSAPULPNN" ]; then
                            echo "Comparing the output..."
                            # Collect output from the log file and compare with the golden output
                            python compareOutput.py --log_file $app_folder/gvsoc.log --MHSA_params $S $E $P $H --kernel_name $kernel_name --app_folder $app_folder

                            # Write profiling data to the log file
                            python extractProfilingData.py --log_file $app_folder/gvsoc.log --MHSA_params $S $E $P $H --kernel_name $kernel_name --test_name $test_name --result_file ./Results/kernelTestResults.log
                        fi
                    fi
                done
            done
        done
    done
done











