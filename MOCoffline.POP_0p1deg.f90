!*******************************************************************
subroutine moczonalint(nyaux,nx,ny,nz,nt,ntr,tlat,lat_aux_grid, &
                         rmlak,work1,wmsg,moc)

      implicit none
      integer, intent(in) ::  nyaux,nx,ny,nz,ntr,nt 
      real*8, intent(in) ::  work1(nx,ny,nz,nt), wmsg
      real*8, intent(out) ::  moc(nyaux,nz,ntr,nt)
      real*8, intent(in) ::     lat_aux_grid(nyaux), tlat(nx,ny)
      integer, intent(in) ::  rmlak(nx,ny,ntr)
      integer ::  iyy,ix,iy,iz,it,ir,icnt1,icnt2,icnt3

! initilize 
      do it=1,nt
       do ir=1,ntr
        do iz=1,nz
         do iyy=1,nyaux
          moc(iyy,iz,ir,it) = 0.0d0 
         end do
        end do
       end do
      end do
!     print *, 'made it into mocloops!'

! globe [note: rmlak(:,:,1)]

      do iyy=2,nyaux
      do it=1,nt
          do iy=1,ny
          do ix=1,nx
             if (tlat(ix,iy).ge.lat_aux_grid(iyy-1) .and. &
                 tlat(ix,iy).lt.lat_aux_grid(iyy)   .and. &
                 rmlak(ix,iy,1).eq.1) then

               do iz=1,nz
                    if (work1(ix,iy,iz,it).ne.wmsg)  then          
        moc(iyy,iz,1,it) = moc(iyy,iz,1,it) + work1(ix,iy,iz,it)
                    end if
               end do
             end if
          end do
          end do
       end do
      end do

! atlantic [note: rmlak(:,:,2)]

      do iyy=2,nyaux
      do it=1,nt
          do iy=1,ny
          do ix=1,nx
             if (tlat(ix,iy).ge.lat_aux_grid(iyy-1) .and. &
                 tlat(ix,iy).lt.lat_aux_grid(iyy)   .and. &
                 rmlak(ix,iy,2).eq.1) then

               do iz=1,nz
                    if (work1(ix,iy,iz,it).ne.wmsg)  then
        moc(iyy,iz,2,it) = moc(iyy,iz,2,it) + work1(ix,iy,iz,it)
                    end if
               end do
             end if
          end do
          end do
       end do
      end do

end subroutine moczonalint
!*******************************************************************

!*******************************************************************
subroutine sig2fluxconv(nt,nz,ny,nx,nsig,kmt,kmu,ztop,dzu,zsigtop, &
             zsigbot,workuNE,workvNE,uout,vout,wout,mval2)
      implicit none
      integer, intent(in) ::  nt,nz,ny,nx,nsig, kmt(nx,ny), kmu(nx,ny)
      real, intent(in) ::    ztop(nz), dzu(nx,ny,nz)
      real, intent(in) ::    zsigtop(nx,ny,nsig,nt),zsigbot(nx,ny,nsig,nt)
      real*8, intent(in) ::  workuNE(nx,ny,nz,nt), workvNE(nx,ny,nz,nt),mval2
      real*8, intent(out) ::  wout(nx,ny,nsig,nt), vout(nx,ny,nsig,nt)
      real*8, intent(out) ::  uout(nx,ny,nsig,nt)
      integer :: it,isig,iz,iy,ix,ij,ixx(4),iyy(4)
      real*8 ::  wgt(nsig),zbsig,ztsig,zb,zt
      real*8 ::  dzsig, sumwgt, udydz,vdxdz, dz
      real*8 ::  fueW,fueE,fvnN,fvnS
      real*8 ::  tmpusig(4,nsig),tmpvsig(4,nsig)

      uout = mval2
      vout = mval2
      wout = mval2

      do it=1,nt
      do iy=2,ny
      do ix=1,nx
!  skip land mask points
        if (kmt(ix,iy).gt.0) then

!  Define indices for the corner U points :  
!  1=NE, 2=NW, 3=SW, 4=SE
          ixx(1) = ix
          ixx(4) = ix
          if (ix.ne.1) then
             ixx(2)=ix-1
             ixx(3)=ix-1
          else
             ixx(2)=nx
             ixx(3)=nx
          end if 
          iyy(1) = iy
          iyy(2) = iy
          iyy(3) = iy-1
          iyy(4) = iy-1

           do ij=1,4
             tmpusig(ij,:) = mval2
             tmpvsig(ij,:) = mval2
            do iz=1,nz
             dz = dzu(ixx(ij),iyy(ij),iz)
             zt = ztop(iz)
             zb = zt+dz
             if (iz.le.kmt(ix,iy)) then
                udydz = workuNE(ixx(ij),iyy(ij),iz,it)
                vdxdz = workvNE(ixx(ij),iyy(ij),iz,it)
             else
                udydz = 0.
                vdxdz = 0.
             end if
             do isig=1,nsig
              zbsig = zsigbot(ixx(ij),iyy(ij),isig,it)
              ztsig = zsigtop(ixx(ij),iyy(ij),isig,it)
              wgt(isig) = 0.0
              if (zbsig.lt.1.e15) then
               dzsig = zbsig-ztsig
! compute wgt() array, which corresponds to the fraction
! of this z-layer flux that is associated with this
! isopycnal layer.  Sum over all sigma layers must give
! total horizontal flux for this z-level!

!  if this z-layer falls entirely within the isopycnal layer:
               if (zb.le.zbsig.and.zt.ge.ztsig) then
                wgt(isig) = 1.0
               end if
!  if only bottom of z-cell is within isopycnal layer:
               if (zb.ge.ztsig.and.zt.lt.ztsig) then
                wgt(isig) = (zb-ztsig)/dz
               end if
!  if only top of z-cell is within isopycnal layer:
               if (zt.le.zbsig.and.zb.gt.zbsig) then
                wgt(isig) = (zbsig-zt)/dz
               end if
!  if the isopycnal layer falls entirely within the z-layer:
               if (zt.lt.ztsig.and.zb.gt.zbsig) then
                wgt(isig) = dzsig/dz
               end if
              end if
             end do

! Use normalized wgt() array to compute tmpusig, tmpvsig
             sumwgt = sum(wgt)
!            if (sumwgt.ne.1.0) then
!              write (6,*) 'sig2fluxconv:: ERROR at i,j,k = ',ix,iy,iz,
!    +      sumwgt
!            end if
             if (sumwgt.ne.0.0) then
               do isig=1,nsig
                if (wgt(isig).ne.0.) then
                 wgt(isig) = wgt(isig)/sumwgt
                 if (tmpusig(ij,isig).eq.mval2) then
                  tmpusig(ij,isig)= wgt(isig)*udydz
                 else
                  tmpusig(ij,isig)=tmpusig(ij,isig)+wgt(isig)*udydz
                 end if
                 if (tmpvsig(ij,isig).eq.mval2) then
                  tmpvsig(ij,isig)= wgt(isig)*vdxdz
                 else
                  tmpvsig(ij,isig)=tmpvsig(ij,isig)+wgt(isig)*vdxdz
                 end if
                end if
               end do
             end if

            end do
           end do
!  This ends the loops over all z and all 4 corners.
!  Now, tmpusig, tmpvsig are defined, so compute
!  convergence and fill vout and wout:
          do isig=1,nsig
           uout(ix,iy,isig,it) = tmpusig(1,isig)
           vout(ix,iy,isig,it) = tmpvsig(1,isig)
           if (all(tmpvsig(:,isig).eq.mval2).and. & 
                all(tmpusig(:,isig).eq.mval2)) then
            wout(ix,iy,isig,it) = mval2
           else
            where(tmpusig(:,isig).eq.mval2) tmpusig(:,isig)=0.
            where(tmpvsig(:,isig).eq.mval2) tmpvsig(:,isig)=0.
            fueE = 0.5*(tmpusig(1,isig)+tmpusig(4,isig))
            fueW = 0.5*(tmpusig(2,isig)+tmpusig(3,isig))
            fvnN = 0.5*(tmpvsig(1,isig)+tmpvsig(2,isig))
            fvnS = 0.5*(tmpvsig(3,isig)+tmpvsig(4,isig))
            wout(ix,iy,isig,it) = -(fueE-fueW+fvnN-fvnS)
           end if
          end do

        end if
      end do
      end do
      end do

end subroutine sig2fluxconv
!*******************************************************************

!*******************************************************************
subroutine sig2z(nt,nz,ny,nx,nsig,popsig,kmu,z_t,dzu, &
            slevst,slevsb,zoutt,zoutb,mval)
      implicit none
      integer, intent(in) :: nt,nz,ny,nx,nsig,kmu(nx,ny)
      real, intent(in) :: mval
      real, intent(in) :: popsig(nx,ny,nz,nt)
      real, intent(in) :: slevst(nsig),slevsb(nsig)
      real, intent(in) :: z_t(nz),dzu(nx,ny,nz)
      real, intent(out) :: zoutt(nx,ny,nsig,nt),zoutb(nx,ny,nsig,nt)
      integer :: it,is,iz,iy,ix,ij,kmax
      real :: dsig,dzdsig,hu,sigmin,sigmax,popsig1d(nz)

      zoutt = mval
      zoutb = mval

      do iy=1,ny
      do ix=1,nx
!  skip land mask points
        if (popsig(ix,iy,1,1).ne.mval) then
          do it=1,nt
           popsig1d = popsig(ix,iy,:,it)
!          kmax = maxloc(popsig1d,1,popsig1d.ne.mval)
           kmax = kmu(ix,iy)
           hu = z_t(kmax) + dzu(ix,iy,kmax)
           sigmin = popsig1d(1)
           sigmax = popsig1d(kmax)

!          if (ix.eq.280.and.iy.eq.150) then
!            write (6,*) 'kmax = ',kmax
!            write (6,*) 'hu = ',hu
!            write (6,*) 'sigmin = ',sigmin
!            write (6,*) 'sigmax = ',sigmax
!            write (6,*) 'mval = ',mval
!            write (6,*) 'popsig(280,150,:,it) = ',popsig1d
!          end if

            do is=1,nsig

! Depth of top of sigma layer (assuming it falls in-between model sigma values):
             if (slevst(is).ge.sigmin.and.slevst(is).le.sigmax) then
              do iz=2,kmax
               if (slevst(is).ge.popsig1d(iz-1).and. &
                 slevst(is).lt.popsig1d(iz)) then
        dsig=slevst(is)-popsig1d(iz-1)
        dzdsig=(z_t(iz)-z_t(iz-1))/(popsig1d(iz)-popsig1d(iz-1))
        zoutt(ix,iy,is,it) = z_t(iz-1)+dzdsig*dsig
               end if
              end do
               if (slevst(is).eq.sigmax) then
        zoutt(ix,iy,is,it) = hu
               end if
             end if

! Depth of bottom of sigma layer (assuming it falls in-between model sigma values):
             if (slevsb(is).ge.sigmin.and.slevsb(is).le.sigmax) then
              do iz=2,kmax
               if (slevsb(is).ge.popsig1d(iz-1).and.slevsb(is).lt.popsig1d(iz)) then
        dsig=slevsb(is)-popsig1d(iz-1)
        dzdsig=(z_t(iz)-z_t(iz-1))/(popsig1d(iz)-popsig1d(iz-1))
        zoutb(ix,iy,is,it) = z_t(iz-1)+dzdsig*dsig
               end if
              end do
               if (slevsb(is).eq.sigmax) then
        zoutb(ix,iy,is,it) = hu
               end if
             end if

! Deal with outcrops (if above was successful).
! 	surface outcrop: zoutb exists but not zoutt
! 	bottom outcrop: zoutt exists but not zoutb
             if (zoutb(ix,iy,is,it).ne.mval.and.zoutt(ix,iy,is,it).eq.mval) then
               zoutt(ix,iy,is,it) = 0.0
             end if
             if (zoutt(ix,iy,is,it).ne.mval.and.zoutb(ix,iy,is,it).eq.mval) then
               zoutb(ix,iy,is,it) = hu
             end if

! If the above blocks failed (did not define zoutt or zoutb), check to see whether
! model {sigmin,sigmax} range falls entirely within this slevs interval:
             if (slevst(is).le.sigmin.and.slevsb(is).ge.sigmax) then
               zoutt(ix,iy,is,it) = 0.0
               zoutb(ix,iy,is,it) = hu
             end if

            end do

! Deal with cases where the slevs range is entirely greater than
! or less than the {sigmin,sigmax} range at an active ocean point:
           if (slevst(1).gt.sigmax) then
              zoutt(ix,iy,1,it) = 0.0
              zoutt(ix,iy,2:nsig,it) = mval
              zoutb(ix,iy,1,it) = hu
              zoutb(ix,iy,2:nsig,it) = mval
           end if
           if (slevsb(nsig).lt.sigmin) then
              zoutt(ix,iy,nsig,it) = 0.0
              zoutt(ix,iy,1:(nsig-1),it) = mval
              zoutb(ix,iy,nsig,it) = hu
              zoutb(ix,iy,1:(nsig-1),it) = mval
           end if

! Deal with outcrops that weren't caught above:
             if (zoutb(ix,iy,1,it).ne.mval) then
               zoutt(ix,iy,1,it) = 0.0
             end if
             if (zoutt(ix,iy,nsig,it).ne.mval) then
               zoutb(ix,iy,nsig,it) = hu
             end if
! Check that full ocean depth is accounted for
           if (.not.any(zoutt(ix,iy,:,it).eq.0.0).or..not.any(zoutb(ix,iy,:,it).eq.hu)) then
             write (6,*) 'sig2z:: ERROR at i,j,hu = ',ix,iy,hu
           if (ix.eq.280.and.iy.eq.331) then
             write (6,*) 'zoutt = ',zoutt(ix,iy,:,it)
             write (6,*) 'zoutb = ',zoutb(ix,iy,:,it)
           end if
           end if

          end do
!       else
!          write (6,*) 'land at i,j = ',ix,iy
        end if
      end do
      end do

end subroutine sig2z
!*******************************************************************
