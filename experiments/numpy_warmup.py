import numpy as np


X = np.array([[1,2,3],[4,5,6]])

print(f"Data \n {X}")
print(f"Shape {X.shape}")
print(f"Rank dimension {X.ndim}")

#shape mistmatch experiment
try: 
    salah = np.array([[1,1], [1,1]])
    print(X  + salah)
except ValueError as e:
    print(f"Error catch {e}")
    
    
    
#Broadcasting experiment
a = 10
b = np.array([1,2,3])
result_boradcasting = a + b
print(f"Result of adding scalar to array: {result_boradcasting}")


X = np.array([[1,2,3],[4,5,6]])
Y = np.array([1,0,1])
result_array = X + Y
print(f"Result of adding 1D array to 2D array: \n{result_array}")


data = np.ones((10,5))
multiplier = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
result_multiplier = data * multiplier
print(f"shape hasil : {result_multiplier}")
print(f"three first row : \{result_multiplier[:3]}")


#Pricing and  Slicing
baris_tiga = result_multiplier[2, :]
print(f"Baris ketiga : {baris_tiga}")

kolom_satu = result_multiplier[:,0]
print(f"kolom satu : {kolom_satu}")

kotak = result_multiplier[0:3,0:3]
print(f"kotak 3x3 awal \n {kotak}")


target_y = result_multiplier[:, -1]
print("last column :")
print(target_y)
print(f"shape target {target_y.shape}")

#The reverse slice 
features_X = result_multiplier[:, :-1]
print(f"shape of features_X : {features_X}")


#Boolean masking
kondisi = result_multiplier > 7

print("Hasil Masking (Boolean):\n", kondisi[:3]) # Cuma liat 3 baris awal

data_filter = result_multiplier[result_multiplier > 7]

print("\nData yang lolos filter (> 7):")
print(data_filter)


#Reshaping column
data_1d = np.arange(12)
kotak = data_1d.reshape(6,2)
print(f"Matrix 3x4 \n {kotak}")

#Tranpose
reshape_result = kotak.reshape(3,4)
transpose_result = reshape_result.T

print(f"reshape result :\n {reshape_result} ")
print(f"Transpose result :\n {transpose_result} ")

#Aggregations & axis
# Case 5 student with 3 subject scores per student
scores = np.array([
    [80, 75, 90],
    [60, 70, 65],
    [95, 90, 100],
    [70, 85, 80],
    [50, 60, 55]
])

#average score perstudent
subject_mean = np.mean(scores, axis=0)
print(subject_mean)


slice_subject = scores[2,:]
print(slice_subject)
student_3_max = np.max(slice_subject, axis=0)
print(student_3_max)