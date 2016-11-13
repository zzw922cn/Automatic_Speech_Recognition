'''
Experiment of a single-layer bidirectional recurrent neural network trained with
connectionist temporal classification to predict character sequences from 39 x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  

author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-11-09
'''

def test1():
    a = [1,2,3,5]
    b = [4,5,6,7]
    for x,y in zip(a,b):
	yield (x,y)

t = test1()

class A():
    def _register(self,name,value):
	for n,v in zip(name,value):
            self.__dict__[n]=v

    def _geter(self,name):
	value = []
	for n in name:
            value.append(getattr(self,n,'None'))
	return value
	
a=A()
a._register(['m1','m2','m3'],[1,2,(1,2)])
v = a._geter(['m1','m2','m3'])
print v


