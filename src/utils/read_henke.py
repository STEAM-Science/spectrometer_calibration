import os
import numpy as np

def getArrays():
    num_energies_to_read = 0
    num_to_read = 0
    energies = []
    f1 = []
    f2 = []
    total_count = 0
    element_count = 0

    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))	
        # Find henke.dat, for Windows users, Windows corrects the fowardslashes to backslashes		
        henke_data = f'{root_dir}/henke_model/henke.dat'
    
        with open(henke_data, 'rb') as f:
            while True:  
                total_count+=1
                combinedBytes = bytearray()
                byte1 = f.read(1)
                if not byte1:
                    break
                else:
                    combinedBytes.append(int.from_bytes(byte1, "big"))

                byte2 = f.read(1)
                if not byte2:
                    break
                else:
                    combinedBytes.append(int.from_bytes(byte2, "big"))


                byte3 = f.read(1)
                if not byte3:
                    break
                else:
                    combinedBytes.append(int.from_bytes(byte3, "big"))

                byte4 = f.read(1)
                if not byte4:
                    break
                else:
                    combinedBytes.append(int.from_bytes(byte4, "big"))
                


                #Here we grab the first two integers of the file and save them into variables so that we read the next 300 integers in the file
                if total_count == 1:
                    num_energies = int.from_bytes(combinedBytes, "big")
                    num_energies_to_read = int.from_bytes(combinedBytes, "big") + total_count + 1
                    print("========== 1st int =================")
                    print(int.from_bytes(combinedBytes, "big"), end="\n\n")

                #2nd integer which is the values we need to alternate between and read into f1 and f2
                if total_count == 2:
                    num_to_read = (num_energies*2) + num_energies_to_read #we multiply by 2 since f1 and f2 each will be of length of 2nd int = 92
                    print("========== 2nd int =================")
                    print(int.from_bytes(combinedBytes, "big"), end="\n\n")

                    print("============== ENERGIES ================", end="\n")


                #Here we will put logic to put energies into an array and also populate f1 and f2 arrays
                if(total_count>2 and total_count <= num_energies_to_read):
                    element_count+=1
                    print("Energy {num_element}".format(num_element = element_count), end=" ")
                    energy = int.from_bytes(combinedBytes, "big")
                    print(energy)
                    energies.append(energy)

                #Here we will read in values into f1 and f2. f1 gets
                if(total_count > num_energies_to_read and total_count <= num_to_read):
                    element_count+=1
                    val = int.from_bytes(combinedBytes, "big")

                    #if it is an even element put it into f2
                    if(element_count % 2 == 0):
                        print("Element {num_element}, Appending {element} to f2".format(num_element = element_count, element = val), end="\n")
                        f2.append(val)
                    else:
                        print("Element {num_element}, Appending {element} to f1".format(num_element = element_count, element = val), end="\n")
                        f1.append(val) #if its odd we put it into f1


                # These are just so we can print and see where the energies and array values end
                if num_energies_to_read != 0 and total_count == num_energies_to_read:
                    element_count = 0
                    print("========== END OF Energies =================\n\n\n\n")

                    print("======= BEGIN TO POPULATE F1 and F2 arrays ===============\n")

                if num_to_read != 0 and total_count == num_to_read:
                    total_count = 3
                    element_count = 0
                    print("============= END OF F1 and F2 ===================\n")

                    #check lengths of arrays
                    # print(energies)
                    # print(f1)
                    # print(f2)
                    return energies, f1, f2
            
    except FileNotFoundError:
        print(f'Could not open file "henke.dat" or {henke_data}')
        raise
            
          
        
        
# energies, f1, f2 = getArrays()

# print("====== ENERGIES =====")
# print(energies)
# print(len(energies))
# print("====== F1 =====")
# print(f1)
# print(len(f1))
# print("====== F2 =====")
# print(f2)
# print(len(f2))