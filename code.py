
# Save data in file
np.savetxt('A_data.dat', data)

# Load data from file
data = np.loadtxt('A_data.dat')


# write file
file = open("data.py","w")  #.py .txt .pdf
file.write('int= %d \r\n' % number)
file.close()

# Read file
f=open("data.py", "r")
f.read()