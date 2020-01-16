
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
import scipy 

def gradx(im):
    "renvoie le gradient dans la direction x"
    imt=np.float32(im)
    gx=0*imt
    gx[:,:-1]=imt[:,1:]-imt[:,:-1]
    return gx

def grady(im):
    "renvoie le gradient dans la direction y"
    imt=np.float32(im)
    gy=0*imt
    gy[:-1,:]=imt[1:,:]-imt[:-1,:]
    return gy
    
def div1(V):#prend le champ de vecteurs
    Vx=V[:,:,0]
    Vy=V[:,:,1]
    ddx=0*Vx
    ddy=ddx
    
    ddx[1:-1,:]=Vx[1:-1,:]-Vx[:-2,:]
    ddx[0,:]=Vx[0,:]
    ddx[-1,:]=-Vx[-2,:]
    
    ddy[:,1:-1]=Vy[:,1:-1]-Vy[:,:-2]
    ddy[:,0]=Vy[:,0]
    ddy[:,-1]=-Vy[:,-2]
    
    div=ddx+ddy
    
    return div
    
def div2(Vx,Vy):#prends les deux champs scalaires
    ddx=0*Vx
    ddy=ddx.copy()
    
    ddx[1:-1,:]=Vx[1:-1,:]-Vx[:-2,:]
    ddx[0,:]=Vx[0,:]
    ddx[-1,:]=-Vx[-2,:]
    
    ddy[:,1:-1]=Vy[:,1:-1]-Vy[:,:-2]
    ddy[:,0]=Vy[:,0]
    ddy[:,-1]=-Vy[:,-2]
    
    div=ddx+ddy
    
    return div

    
def laplacien(im):
    return div2(gradx(im),grady(im))


def filtergauss(im):
    """applique un filtre passe-bas gaussien. coupe approximativement a f0/4"""
    (ty,tx)=im.shape
    imt=np.float32(im.copy())
    pi=np.pi
    XX=np.concatenate((np.arange(0,tx/2+1),np.arange(-tx/2+1,0)))
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    
    YY=np.concatenate((np.arange(0,ty/2+1),np.arange(-ty/2+1,0)))
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
    # C'est une gaussienne, dont la moyenne est choisie de sorte que
    # l'integrale soit la meme que celle du filtre passe bas
    # (2*pi*sig^2=1/4*x*y (on a suppose que tx=ty))
    sig=(tx*ty)**0.5/2/(pi**0.5)
    mask=np.exp(-(XX**2+YY**2)/2/sig**2)
    
    #print (mask)
    imtf=np.fft.fft2(imt)*mask
    return np.real(np.fft.ifft2(imtf)) 



def replaceRectangle(imSource, patch, rectangles):
    """
    remplace une zone rectangulaire
    définie par les coordonnées de deux coins par une autre
     """
    imSourceMod = imSource.copy()
    for rect in rectangles:
        imSourceMod[rect[0]: rect[2],rect[1]: rect[3] ] = patch[rect[0]: rect[2],rect[1]: rect[3] ]
    return(imSourceMod)

def blurVertically(imSource, patch, n):
    imSourceMod = imSource.copy()

    l = []
    sizeI = imSource.shape[0]
    sizeJ = imSource.shape[1]
    print(sizeI)
    sliceJ =int(sizeJ /n) 
    for i in range (0,n):
        j1 = i * sliceJ
        j2 = j1 + sliceJ
        l.append(replaceRectangle(imSource, patch, [(0, j1, sizeI, j2)] ))
    return (l)



def targetGradient2(G, gradList, ponderations ):
    """ici gradlist est une liste de tuples contenant les deus images de gradients (en x et en y) plutot qu'une liste de vecteurs gradients"""
    (sizeI, sizeJ) = G.shape[0:2] 
    eigenValuesPlus = np.zeros(G.shape[0:2])
    eigenVectorsPlus = np.zeros( (sizeI,sizeJ,2) )
    targetGrad = np.zeros(G.shape[0:3])

    n = len(ponderations)

    for i in range (0,sizeI):
        for j in range (0, sizeJ):
            values, vectors = np.linalg.eig(G[i][j])
            indexPlus = values.argsort()[::-1]

            eigenValuesPlus[i][j], eigenVectorsPlus[i][j] = values[indexPlus[0]], vectors[indexPlus[0]]

            #we compute the expression whose we take the sign
            v = 0
            s = 0
            for k in range (0, n):
                gradVector=np.array(gradList[k][0][i,j],gradList[k][1][i,j])
                s += ponderations[k][i,j]*gradVector
            u =  np.dot(np.transpose(eigenVectorsPlus[i][j]) , s )

            sign = np.sign(u)

            v = np.sqrt(eigenValuesPlus[i][j])*eigenVectorsPlus[i][j]* sign
            targetGrad[i][j] = v
    
    return (targetGrad)


def grad(im):
    """retourne un couple d'images des coordonnées x et y du gradient de im"""
    (tx,ty)=im.shape
    gradxx=gradx(im)
    gradyy=grady(im)
    grad=[]
    for ligne in range(tx):
        gradligne=[]
        for col in range(ty):
            gradligne.append(np.array([gradxx[ligne,col],gradyy[ligne,col]]))
        grad.append(gradligne)
    grad=np.asarray(grad)
    return grad

def grad2(im):
    """param : image   sortie : deux images de grad (selon x et selon y)"""
    return (gradx(im),grady(im))

def normGrad(grad):
    """prend un tableau représentant le gradient en parametre"""
    shape = grad.shape
    normgrad = np.zeros(shape[0:2])
    for i in range (0, shape[0]):
        for j in range (0, shape[1]):
            normgrad[i][j] = np.linalg.norm(grad[i][j])
    return(normgrad)

def normGrad2(im):
    """param : image   sortie : image de la norm du gradient de l'image"""
    grad=grad2(im)
    return (grad[0]**2+grad[1]**2)**0.5

def normGrad_list2(im_list):
    return [normGrad2(im) for im in im_list]

def grad_list2(im_list):
    """ sortie : liste de tuples représentant les gradients des images données en entrée"""
    return [grad2(im) for im in im_list]


def salency_weights2(NormGrad_list):
    """ liste des tableaux de normes de gradients en entrée"""
    (ti,tj)=NormGrad_list[0].shape
    N=len(NormGrad_list)
    s=[np.zeros((ti,tj)) for k in range(N)]
    #on applique la formule des poids
    for i in range(ti):
        for j in range(tj):
            denom=np.linalg.norm([NormGrad_list[k][i,j] for k in range(N)])
            if denom==0:
                for n in range(N):
                    s[n][i,j]=0
            else:
                for n in range(N):
                    s[n][i,j]=NormGrad_list[n][i,j]/denom
    return s

def structureMatrix2(Grad):
    """argument : un tuple représentant les deux images de gradients (x et y) d'une image"""
    (ti,tj)=Grad[0].shape
    G=np.zeros((ti,tj,2,2))
    G[:,:,0,0]=Grad[0]**2
    G[:,:,1,1]=Grad[1]**2
    G[:,:,0,1]=G[:,:,1,0]=Grad[0]*Grad[1]
    return G
    
    

    

def wMatrix(shape, sigma):
    """retourne une gausienne 2D"""
    (tx, ty) = shape
    XX = np.arange(-tx//2+1,tx//2+1)
    XX=np.ones((ty,1))@(XX.reshape((1,tx)))
    YY = np.arange(-ty//2+1,ty//2+1)
    YY=(YY.reshape((ty,1)))@np.ones((1,tx))
        
    mask=np.exp(-(XX**2+YY**2)/2/sigma**2) / (np.sqrt(2*np.pi)*sigma)
    
    return( mask)



def jFunction(k, alpha,r):
    return (k*np.arctan(alpha*r))

    
#Prend une petite zone 5x5 autour de du point I(i,j), pad de zeros en dehors de I
def partPad(I, i, j):
    ret = np.zeros((5,5))
    shape = I.shape
    
    a = max(i-2,0)
    b = min(i+3, shape[0])
    
    c = max(j-2,0)
    d = min(j+3, shape[1])
    
    ret[2-(i-a): 2 + (b-i), 2-(j-c): 2+(d-j)] = I[a:b,c:d]
    return(ret)
    
def Rterm(I, kJ, alpha):
    
    shape = I.shape
    ret = np.zeros(shape)
    sigma = 1.0
    
    testR = np.zeros(shape)
    
    #on crée une gausienne sur 5x5
    wPetit = wMatrix((5,5), sigma)
    wPetit = wPetit/wPetit.sum()
    print("debut R")
    #on parcours tous les points
    for i in range (0, shape[0]):
        for j in range (0, shape[1]):
                         
            morceau = partPad(I, i,j)      
            morceau = I[i][j] - morceau

            #on applique J' a la petite zone
            for k in range(0,morceau.shape[0]):
                for l in range(0,morceau.shape[1]):
                    morceau[k][l] = jFunction(kJ,alpha, morceau[k][l] )
            #on multiplie par la guassienne
            testR[i,j] = 2*(wPetit*morceau).sum()
 
    return testR
    


def multistructureMatrix2(Grad_list,s):
    """argument : liste de tuples d'images de gradients"""
    N=len(Grad_list)
    (ti,tj)=Grad_list[0][0].shape
    Gs=np.zeros((ti,tj,2,2))
    for n in range(N):
        Gn=structureMatrix2(Grad_list[n])
        for i in range(ti):
            for j in range(tj):
                Gs[i,j,:,:]+=s[n][i,j]**2*Gn[i,j,:,:]
    return Gs

    
def imLadjal(V,imList):
    mongradx=V[:,:,0]
    mongrady=V[:,:,1]
    fft2=np.fft.fft2
    ifft2=np.fft.ifft2
    lapla=np.zeros(imList[0].shape)
    lapla[0,0]=-4
    lapla[0,1]=1
    lapla[1,0]=1
    lapla[-1,0]=1
    lapla[0,-1]=1
    flapla=fft2(lapla)
    flapla[0,0]=1
    
    
    divgrad= div2(mongradx,mongrady)
    fdivV=-fft2(divgrad)
    fx=fdivV/flapla
    fx[0,0]=0

    x=np.real(-ifft2(fx))
    x+=(sum(im for im in imList)).mean()/len(imList)
    
    return(x)
    
    
#%%

LISTER = []
IMAGES = []
print(LISTER)
def descenteGrad(sigma,k,alpha,beta,gamma,eta,delta,seuil,I_init,I0,V):
    divV=div1(V)

    I=I_init            
       
    IMAGES.append(I)
    
    #on itère le procédé
    for iteration in range(seuil):
        
        R = Rterm(I, k, alpha)
        LISTER.append(R)

        I=I*(1-2*(beta+gamma)*delta)+  2*eta*(laplacien(I)-divV)*delta + (beta*np.ones(I0.shape)+2*gamma*I0+ R)*delta
        #troncatures
        I[I>1] = 1.0
        I[I<0] = 0.0

        IMAGES.append(I)
        scipy.misc.imsave("inter_"+str(iteration)+".png", I)
        
    return I

#%%


#The input images are loaded into a list
directoryProject = '' 

im0=skio.imread(directoryProject +'lena0.gif')
im1=skio.imread(directoryProject +'lena1.gif')
im2=skio.imread(directoryProject +'lena2.gif')
im3=skio.imread(directoryProject +'lena3.gif')
im4=skio.imread(directoryProject +'lena4.gif')

imLena =skio.imread('C:\\Users\\Choeb\\Desktop\\projetIMA_fusion\\IMA_fusion_asserpe_zafar\\images\\lena.gif')

im_list=[im0,im1,im2,im3,im4]
#normalisation
im_list = [x/255.0 for x in np.float32(im_list)]


#%%
visu = True


#CAlcul des données intermédiaires nécessaires
GradList = grad_list2(im_list)
NormGradList = normGrad_list2(im_list)
Weights = salency_weights2(NormGradList)
G = multistructureMatrix2(GradList,Weights )
V = targetGradient2(G, GradList, Weights)
Vnorm = normGrad(V)

im_mean = sum([1/len(im_list)*im_list[k] for k in range(len(im_list))])

#points de départs possibles pour la descente de gradient
im_init = sum([Weights[k]*im_list[k] for k in range(len(im_list))])
imSaid = imLadjal(V,im_list)


#%%

#Descente de gadient
imFin = descenteGrad(1.0, 0.1, 10, 0.5, 0.3, 0.1, 0.1, 50, im_init, im_mean , V )


#%%

for i in range(-10,-1 ):
    plt.hist(IMAGES[i].ravel(),256,[0,1])
    plt.savefig("hist" + str(i)+".png")
    plt.clf()
