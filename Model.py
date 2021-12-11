import numpy as np

#####IMPLEMENTATION OF LINEAR REGRESSION USING GRADIENT DESCENT###########
class LinearRegression:
  def __init__(self):
    pass
  
  def getThetas(self):
    return self.theta0, self.thetas

  def getCostsAtAllIterations(self):
    return self.costsAtAllIterations

  def predict(self, test_X):
    predicted_Y=np.dot(test_X, self.thetas)+self.theta0
    return predicted_Y

  def fit(self, train_X, train_Y, learningRate=0.02, numOfIterations=2000, validation_X=None, validation_Y=None, printCost=True, storeCosts=True):

    self.thetas=np.zeros((train_X.shape[1],1)) # all thetas from theta1, theta2,...thetaN where N is number of dimensions of each data item
    self.theta0=0.0 #theta0, i.e. the bias term
    m=train_X.shape[0] #number of training examples
    
    if storeCosts is True:
      self.costsAtAllIterations=np.zeros(shape=(numOfIterations,2)) #used for plotting the cost w.r.t number of iterations
    else:
       self.costsAtAllIterations=None

    for i in range(numOfIterations):
      #forward pass
      predicted_Y=np.dot(train_X, self.thetas)+self.theta0

      #computing cost
      residuals= np.subtract(predicted_Y, train_Y)
      squaredResiduals=np.square(residuals)
      sumOfSquaredResiduals=np.sum(squaredResiduals, axis=0)
      sumOfSquaredResiduals= np.squeeze(sumOfSquaredResiduals)
      meanSquaredResiduals=sumOfSquaredResiduals/(2*m)

      #printing cost
      if printCost is True:
        print("Iteration No: "+str(i+1)+". Cost: "+str(meanSquaredResiduals) )
      
      if storeCosts is True:
        self.costsAtAllIterations[i]=[i, meanSquaredResiduals]

      #computing gradient for theta 0 (bias)
      sumOfResiduals=np.sum(residuals, axis=0)
      sumOfResiduals= np.squeeze(sumOfResiduals)
      dtheta0=sumOfResiduals/m
      
      #updating theta0
      self.theta0 = self.theta0 - learningRate * dtheta0

      #computing gradients for all other thetas
      dthetas=np.sum(np.dot(np.transpose(train_X), residuals), axis=0)/m

      #updting all other thetas
      self.thetas=self.thetas-learningRate*dthetas


###IMPLEMENTATION OF LINEAR REGRESSION USING NORMAL EQUATIONS#############
class NormalEquationLinearRegression:
  def fit(self, trainX, trainY):
    m=trainX.shape[0] #number of training examples
    trainX=np.hstack((np.ones((m, 1)), trainX))
    trainXTranspose= np.transpose(trainX);
    trainX_DOT_trainXTranspose_INVERSE= np.linalg.inv(np.dot(trainXTranspose, trainX))
    self.thetas= np.dot(np.dot(trainX_DOT_trainXTranspose_INVERSE, trainXTranspose), trainY)

  def predict(self, testX):
    m=testX.shape[0] #number of test examples
    testX=np.hstack((np.ones((m, 1)), testX))
    predictedY=np.dot(testX, self.thetas)
    return predictedY