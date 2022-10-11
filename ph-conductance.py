import yaml
import numpy as np
import matplotlib.pyplot as plt

hbar = 1.05457182e-34
kB = 1.380649e-23
bohr = 0.529177249

def dBEdT(w,T):
    return (np.exp(hbar*w/kB/T)-1)**-2*np.exp(hbar*w/kB/T)*hbar*w/kB/T**2 

class conductance:

    def __init__(self,fname):
        with open(fname,"r") as stream:
            contents = yaml.safe_load(stream)
            self.nk = contents['nqpoint']
            self.nw = contents['natom']*3
            self.cell = np.array(contents['lattice'])*bohr # the unit in QE is bohr
            self.recivec = np.array(contents['reciprocal_lattice'])/bohr
            self.vol = np.abs(np.dot(np.cross(self.cell[0,:],self.cell[1,:]),self.cell[2,:]))
            self.omega = np.zeros([self.nk,self.nw])
            self.velocity = np.zeros([self.nk,self.nw,3])
            self.kpoints = np.zeros([self.nk,3])
            self.weight = np.zeros([self.nk,])
            for i in range(self.nk):
                self.kpoints[i,:] = np.array(contents['phonon'][i]['q-position']) 
                self.weight[i] = contents['phonon'][i]['weight']
                for j in range(self.nw): 
                    # frequency in unit of THz 
                    self.omega[i,j] = contents['phonon'][i]['band'][j]['frequency']
                    # velocity in unit of Bohr THz (QE). For VASP, it is Ang THz 
                    self.velocity[i,j,:] = np.array(contents['phonon'][i]['band'][j]['group_velocity'])*1e-10*1e12*bohr

            #print(np.linalg.norm(self.velocity[0,2,:]))
            #print(np.linalg.norm(self.velocity[0,1,:]))
            #print(np.linalg.norm(self.velocity[0,0,:]))

            self.total_weight = np.sum(self.weight)
    def cal_T(self,omega_max,nomegaj,direction_index,sigma):
        A = 1
        direction = np.dot(direction_index,self.cell)
        # normalized direction vector
        direction = direction/np.sqrt(direction[0]**2+direction[1]**2+direction[2]**2)
        self.omegaj = np.linspace(0,omega_max,nomegaj)
        self.J = np.zeros([nomegaj,])
        for w in range(nomegaj):
            print(w/nomegaj)
            for ik in range(self.nk):
                for b in range(self.nw):
                    self.J[w] += 0.5*abs(np.dot(self.velocity[ik,b,:],direction))*np.exp(-(self.omegaj[w]-self.omega[ik,b])**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)/self.vol*A*(2*np.pi)/self.total_weight*1e30/1e12/2/np.pi

    def cal_cond(self,J,temperatures):
        self.cond = np.zeros([len(temperatures),])
        for iT in range(len(temperatures)):
            intgd = np.zeros([len(self.omegaj),]) 
            intgd[1:] = hbar*(self.omegaj[1:]*1e12*2*np.pi)*dBEdT(self.omegaj[1:]*1e12*2*np.pi,temperatures[iT])*J[1:]/np.pi/2 
            self.cond[iT] = np.trapz(intgd,self.omegaj*1e12*2*np.pi)
    def cal_dmm(self,j2):
        self.Jdmm = np.zeros([len(self.J),]) 
        for w in range(len(self.omegaj)):
            self.Jdmm[w] = self.J[w]*j2[w]/(self.J[w]+j2[w]) 
    def cal_EL(self,j2):
        self.JEL = np.zeros([len(self.J),])
        for w in range(len(self.omegaj)):
            if self.J[w]<j2[w]:
                # tau = 1
                self.JEL[w] = self.J[w]
            else:
                # tau = 0
                self.JEL[w] = j2[w]
        

sigma = 0.2
wmax = 16
nw = 100
temp = np.linspace(10,800,500)
g1 = conductance("si_mesh10.yaml")
g1.cal_T(wmax,nw,[0,0,1],sigma)
g1.cal_cond(g1.J,temp)
g2 = conductance("al_mesh10.yaml")
g2.cal_T(wmax,nw,[0,0,1],sigma)
g2.cal_cond(g2.J,temp)

plt.figure(1)
plt.plot(g1.omegaj,g1.J)
plt.plot(g2.omegaj,g2.J)
plt.figure(2)
plt.plot(temp,g2.cond,label='Al, perfect')
plt.plot(temp,g1.cond,label='Si, perfect')
g0 = g2.cond
g2.cal_dmm(g1.J)
g2.cal_cond(g2.Jdmm,temp)
plt.plot(temp,1/(1/g2.cond-0.5*(1/g1.cond+1/g0)),label='DMM')
g2.cal_EL(g1.J)
g2.cal_cond(g2.JEL,temp)
plt.plot(temp,1/(1/g2.cond-0.5*(1/g1.cond+1/g0)),label='Elastic limit')
gIEL=np.zeros([len(temp),])
for it in range(len(temp)):
    if g0[it]<g1.cond[it]:
        gIEL[it] = g0[it]
    else:
        gIEL[it] = g1.cond[it]

plt.plot(temp,1/(1/gIEL-0.5*(1/g1.cond+1/g0)),label='Inelastic limit')
plt.ylim([0,3e9])
plt.legend()


plt.xlabel('Temperature (K)')
plt.ylabel('Interfacial thermal conductance (W/m^2/K)')

#plt.savefig('test.pdf')

plt.show()

#print(g1.vol)
#print(g1.omega[0,:])
