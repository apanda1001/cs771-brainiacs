import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def createFeatures( X ):
	# Create features for the training data
    return X

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response

	R = 64
	S1 = 4
	S2 = 4

	# Split the training data into challenges, select bits and responses
	X_trn = Z_train[:, :R]
	p = Z_train[:,R:R+S1]
	q = Z_train[:,R+S1:R+S1+S2]
	Y_trn = Z_train[:,-1]

	train_dict = {}	# Dictionary to store the training data for each xorro

	num_xorro = 2**S1	# S1==S2 so num_xorro = 2**S2

	# Create a dictionary of training data for each xorro
	for a in range(num_xorro):
		for b in range(num_xorro):
			if a==b:	
				continue
			train_mat = np.array([])	# Create an empty training matrix

			# Iterate through the training data
			for i in range(Z_train.shape[0]):

				# Convert the select bits to decimal
				p_dec = 8*int(p[i,0]) + 4*int(p[i,1]) + 2*int(p[i,2]) + int(p[i,3])
				q_dec = 8*int(q[i,0]) + 4*int(q[i,1]) + 2*int(q[i,2]) + int(q[i,3])

				# If the select bits match the xorro, add the challenge response pair to the training data
				if p_dec == a and q_dec == b:

					crp  = (np.append(X_trn[i],np.array([Y_trn[i]])))	# Create a challenge response pair
					crp = crp.reshape(1,-1)	
					if train_mat.size == 0:	
						train_mat = crp	
					else:
						train_mat = np.append(train_mat, crp, axis=0)	# Append the crp to the training matrix
			
			train_dict[(a,b)] = train_mat	# Store the training data for each xorro

	# Train a model for each xorro
	models = {}
	for a in range(num_xorro):
		for b in range(num_xorro):
			if a!=b:	
				train_data = train_dict[(a,b)]	# Get the training data for the xorro
				model = LinearSVC(loss='hinge')		# Create a model
				model.fit(createFeatures(train_data[:,:64]), train_data[:,-1])	# Train the model
				models[(a,b)] = model	# Store the model

	return models					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, models):
################################
#  Non Editable Region Ending  #
################################
	# Use this method to make predictions on test challenges
	R = 64
	S1 = 4
	S2 = 4

	# Split the test data into challenges, select bits and responses
	X_test = X_tst[:, :R]
	p = X_tst[:,R:R+S1]
	q = X_tst[:,R+S1:R+S1+S2]

	# Predict the response for each challenge
	Y_pred = np.zeros(len(X_tst))
	for i in range(X_tst.shape[0]):

		# Convert the select bits to decimal
		p_dec = 8*int(p[i,0]) + 4*int(p[i,1]) + 2*int(p[i,2]) + int(p[i,3])
		q_dec = 8*int(q[i,0]) + 4*int(q[i,1]) + 2*int(q[i,2]) + int(q[i,3])

		# Get the model for the xorro
		model = models[(p_dec,q_dec)]

		# Predict the response
		Y_pred[i] = model.predict(createFeatures(X_test[i].reshape(1,-1)))
    	
	return Y_pred
