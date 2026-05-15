{ Kubo problem using Stokes Einstein}
 

 
variables
  f(1e-7)
  
SELECT ERRLIM=1e-7

{REMATRIX=ON}

COORDINATES cartesian2
 
definitions
  {Define lambda = lam = 2 Lx, eps= kap/lam, en = v}
  
  
  kap =0.003
  
  
  
  
  Dlj=0.3119
  
  lam =1
  Lx = lam   Ly =20*lam
  en= 3
  eps= kap/lam^2
  {Use units where mu11=1}
  m11= 2.000290317898098
  m22=  2.000290317898098
  m12= 1.9997096821019014
 

 
  phi = en/1*(1-cos(2*Pi*x/Lx))
  peq = exp(-phi -y^2/2)

  
  Initial Values
  f = 0
 

equations
 
  f:  m11*dx(dx(f)+ dx(phi)*f) +m12*(2*dx(dy(f))+ dx(phi)*dy(f)+ y*dx(f))/sqrt(eps)+ m22*dy(dy(f) + y*f)/eps=m11*dx(peq)+m12*dy(peq)/sqrt(eps)
 { g:  eps*m11*dx(dx(g)+ dx(phi)*g) +m12*(2*dx(dy(g))+ dx(phi)*dy(g)+ y*dx(g))*sqrt(eps)+ m22*dy(dy(g) + y*g)= m12*dy(peq)}
    constraints
  integral(f)=0

 
boundaries
  region 1
    start(-Lx/2,-Ly/2)
     natural(f)=0
      line to (Lx/2,-Ly/2)
    periodic(x-2*Lx/2,y)
              line to (Lx/2,Ly/2)
 
    natural(f)=0
              line to(-Lx/2,Ly/2)
      line to close
 
 

 
plots
  
  contour(f)  painted

  report(kap,eps, m11- integral((f)*(m11*dx(phi) + m12*y/sqrt(eps)))/integral(peq))
  {report(kap)}
  {report((m11- integral((f)*(m11*dx(phi) + m12*y/sqrt(eps)))/integral(peq))/Dlj-1)}
  
  
  
  
 
end