!*******************************************************************
subroutine wzonalsum(nyaux,nx,ny,nz,nt,ntr,tlat,lat_aux_grid, &
                         rmlak,wflux,wmsg,wfluxzonsum)
!
! Routine to compute zonal sum of wflux for computing MOC using
! the w-method: 
! Psi(y,z) = SUM(y_s,y: SUM(x_w,x_e: wflux(x,y,z) ))
! This routine returns the inner zonal sum.
!
! Input:
!	wflux(x,y,z,t) =w*dx*dy  (m^3/s)
!          where x,y is model T-grid
!
! Output:
!	wfluxzonsum(y2,z,nr,t)  (m^3/s)
!   	   where y2 is the meridional grid for MOC
!                nr is a region dimension for (Global, Atlantic)
!
      implicit none
      integer, intent(in) ::  nyaux,nx,ny,nz,ntr,nt 
      real*8, intent(in) ::  wflux(nx,ny,nz,nt), wmsg
      real*8, intent(out) ::  wfluxzonsum(nyaux,nz,ntr,nt)
      real*8, intent(in) ::     lat_aux_grid(nyaux), tlat(nx,ny)
      integer, intent(in) ::  rmlak(nx,ny,ntr)
      integer ::  iyy,ix,iy,iz,it,ir,icnt1,icnt2,icnt3

! initilize 
      do it=1,nt
       do ir=1,ntr
        do iz=1,nz
         do iyy=1,nyaux
          wfluxzonsum(iyy,iz,ir,it) = 0.0d0 
         end do
        end do
       end do
      end do

! globe [note: rmlak(:,:,1)]

      do iyy=2,nyaux
      do it=1,nt
          do iy=1,ny
          do ix=1,nx
             if (tlat(ix,iy).ge.lat_aux_grid(iyy-1) .and. &
                 tlat(ix,iy).lt.lat_aux_grid(iyy)   .and. &
                 rmlak(ix,iy,1).eq.1) then

               do iz=1,nz
                    if (wflux(ix,iy,iz,it).ne.wmsg)  then          
        wfluxzonsum(iyy,iz,1,it) = wfluxzonsum(iyy,iz,1,it) + wflux(ix,iy,iz,it)
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
                    if (wflux(ix,iy,iz,it).ne.wmsg)  then
        wfluxzonsum(iyy,iz,2,it) = wfluxzonsum(iyy,iz,2,it) + wflux(ix,iy,iz,it)
                    end if
               end do
             end if
          end do
          end do
       end do
      end do

end subroutine wzonalsum
!*******************************************************************

!*******************************************************************
subroutine fluxconv(nt,nz,ny,nx,targnz,kmt,ztop,zbot,dz,targztop, &
             targzbot,workuNE,workvNE,uout,vout,wout,mval)
!
! Routine to compute vertical volume flux given model
! horizontal fluxes on U-grid: workuNE,workvNE. The horizontal fluxes
! are first partitioned from model z-grid (ztop,zbot,dz) to a new target 
! z-grid (targztop,targzbot) before applying model divergence operator.
! This gives dwflux/dz on model T-grid, which is then integrated in
! vertical to give wflux on (T-grid, targztop).
!
! Outputs:
!	uout:  grid-oriented zonal volume flux on new z-grid
!	vout:  grid-oriented meridional volume flux on new z-grid
!	wout:  vertical volume flux on new z-grid
!
      implicit none
      integer, intent(in) ::  nt,nz,ny,nx,targnz, kmt(nx,ny)
      real, intent(in) ::    ztop(nz), zbot(nz), dz(nz)
      real, intent(in) ::    targztop(nx,ny,targnz,nt),targzbot(nx,ny,targnz,nt)
      real*8, intent(in) ::  workuNE(nx,ny,nz,nt), workvNE(nx,ny,nz,nt),mval
      real*8, intent(out) ::  wout(nx,ny,targnz,nt), vout(nx,ny,targnz,nt)
      real*8, intent(out) ::  uout(nx,ny,targnz,nt)
      integer :: it,iz2,iz,iy,ix,ij,ixx(4),iyy(4)
      real*8 ::  wgt(targnz),zbsig,ztsig,zb,zt
      real*8 ::  dzsig, sumwgt, udydz,vdxdz
      real*8 ::  fueW,fueE,fvnN,fvnS
      real*8 ::  tmpu(4,targnz),tmpv(4,targnz)

!     uout = mval
!     vout = mval
      uout = 0.0
      vout = 0.0
      wout = 0.0

      do it=1,nt
      do iy=2,ny
      do ix=1,nx
!  skip land mask points
!       if (kmt(ix,iy).ne.0) then
! edited below for 0.1 grid (04/16/2014)
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
             tmpu(ij,:) = mval
             tmpv(ij,:) = mval
            do iz=1,nz
             zb = zbot(iz)
             zt = ztop(iz)
!            if (iz.le.kmt(ix,iy)) then
                udydz = workuNE(ixx(ij),iyy(ij),iz,it)
                vdxdz = workvNE(ixx(ij),iyy(ij),iz,it)
!            else
!               udydz = 0.
!               vdxdz = 0.
!            end if
             do iz2=1,targnz
              zbsig = targzbot(ixx(ij),iyy(ij),iz2,it)
              ztsig = targztop(ixx(ij),iyy(ij),iz2,it)
              wgt(iz2) = 0.0
              if (zbsig.lt.1.e15) then
               dzsig = zbsig-ztsig
! compute wgt() array, which corresponds to the fraction
! of this z-layer flux that is associated with this
! isopycnal layer.  Sum over all sigma layers must give
! total horizontal flux for this z-level!

!  if this z-layer falls entirely within the isopycnal layer:
               if (zb.le.zbsig.and.zt.ge.ztsig) then
                wgt(iz2) = 1.0
               end if
!  if only bottom of z-cell is within isopycnal layer:
               if (zb.ge.ztsig.and.zt.lt.ztsig) then
                wgt(iz2) = (zb-ztsig)/dz(iz)
               end if
!  if only top of z-cell is within isopycnal layer:
               if (zt.le.zbsig.and.zb.gt.zbsig) then
                wgt(iz2) = (zbsig-zt)/dz(iz)
               end if
!  if the isopycnal layer falls entirely within the z-layer:
               if (zt.lt.ztsig.and.zb.gt.zbsig) then
                wgt(iz2) = dzsig/dz(iz)
               end if
              end if
             end do

! Use normalized wgt() array to compute tmpu, tmpv
             sumwgt = sum(wgt)
!            if (sumwgt.ne.1.0) then
!              write (6,*) 'fluxconv:: ERROR at i,j,k = ',ix,iy,iz,
!    +      sumwgt
!            end if
             if (sumwgt.ne.0.0) then
               do iz2=1,targnz
                if (wgt(iz2).ne.0.) then
                 wgt(iz2) = wgt(iz2)/sumwgt
                 if (tmpu(ij,iz2).eq.mval) then
                  tmpu(ij,iz2)= wgt(iz2)*udydz
                 else
                  tmpu(ij,iz2)=tmpu(ij,iz2)+wgt(iz2)*udydz
                 end if
                 if (tmpv(ij,iz2).eq.mval) then
                  tmpv(ij,iz2)= wgt(iz2)*vdxdz
                 else
                  tmpv(ij,iz2)=tmpv(ij,iz2)+wgt(iz2)*vdxdz
                 end if
                end if
               end do
             end if

            end do
           end do
!  This ends the loops over all z and all 4 corners.
!  Now, tmpu, tmpv are defined, so compute
!  convergence and fill vout and wout:
          do iz2=1,targnz
           uout(ix,iy,iz2,it) = tmpu(1,iz2)
           vout(ix,iy,iz2,it) = tmpv(1,iz2)
           if (all(tmpv(:,iz2).eq.mval).and. & 
                all(tmpu(:,iz2).eq.mval)) then
            wout(ix,iy,iz2,it) = 0.0
           else
            where(tmpu(:,iz2).eq.mval) tmpu(:,iz2)=0.
            where(tmpv(:,iz2).eq.mval) tmpv(:,iz2)=0.
            fueE = 0.5*(tmpu(1,iz2)+tmpu(4,iz2))
            fueW = 0.5*(tmpu(2,iz2)+tmpu(3,iz2))
            fvnN = 0.5*(tmpv(1,iz2)+tmpv(2,iz2))
            fvnS = 0.5*(tmpv(3,iz2)+tmpv(4,iz2))
            wout(ix,iy,iz2,it) = -(fueE-fueW+fvnN-fvnS)
           end if
          end do

          !  vertical sum
          do iz2=1,targnz-1
            wout(ix,iy,targnz-iz2,it) = wout(ix,iy,targnz-iz2,it) + &
                 wout(ix,iy,targnz-iz2+1,it)
          end do
          !  mask out where appropriate
          do iz2=1,targnz-1
            if (all(tmpv(:,iz2).eq.mval).and. &
                all(tmpu(:,iz2).eq.mval)) then
              wout(ix,iy,iz2,it) = mval
            end if
          end do

        end if
      end do
      end do
      end do

end subroutine fluxconv
!*******************************************************************

!*******************************************************************
subroutine sgsfluxconv(nt,nz,ny,nx,targnz,kmt,ztop,zbot,dz &
            ,targztopu,targztopv,targzbotu,targzbotv,worku,workv &
            ,uout,vout,wout,mval)
      implicit none
      integer, intent(in) :: nt,nz,ny,nx,targnz, kmt(nx,ny)
      real, intent(in) :: ztop(nz), zbot(nz), dz(nz)
      real, intent(in) :: targztopu(nx,ny,targnz,nt),targztopv(nx,ny,targnz,nt)
      real, intent(in) :: targzbotu(nx,ny,targnz,nt),targzbotv(nx,ny,targnz,nt)
      real*8, intent(in) :: worku(nx,ny,nz,nt), workv(nx,ny,nz,nt),mval
      real*8, intent(out) :: wout(nx,ny,targnz,nt), vout(nx,ny,targnz,nt)
      real*8, intent(out) ::  uout(nx,ny,targnz,nt)
      integer :: it,iz2,iz,iy,ix,ij,ixx(2),iyy(2)
      real*8 ::  wgtu(targnz),wgtv(targnz)
      real*8 ::  zbsigu,ztsigu,zbsigv,ztsigv,zb,zt
      real*8 ::  dzsig, sumwgtu, sumwgtv, udydz,vdxdz
      real*8 ::  fueW,fueE,fvnN,fvnS
      real*8 ::  tmpu(2,targnz),tmpv(2,targnz)

      uout = 0.
      vout = 0.
      wout = 0.

      do it=1,nt
      do iy=2,ny
      do ix=1,nx
!  skip land mask points
!       if (kmt(ix,iy).ne.0) then
! edited below for 0.1 grid (04/16/2014)
        if (kmt(ix,iy).gt.0) then

!  Define indices for the SGS U points that will contribute to 
!  W at this T-point:  
!  UISOP,USUBM:   1=East face, 2=West face
!  VISOP,VSUBM:   1=North face, 2=South face
          ixx(1) = ix
          if (ix.ne.1) then
             ixx(2)=ix-1
          else
             ixx(2)=nx
          end if 
          iyy(1) = iy
          iyy(2) = iy-1

           do ij=1,2
             tmpu(ij,:) = mval
             tmpv(ij,:) = mval
            do iz=1,nz
             zb = zbot(iz)
             zt = ztop(iz)
             if (iz.le.kmt(ix,iy)) then
                udydz = worku(ixx(ij),iyy(1),iz,it)
                vdxdz = workv(ixx(1),iyy(ij),iz,it)
             else
                udydz = 0.
                vdxdz = 0.
             end if
             do iz2=1,targnz
              zbsigu = targzbotu(ixx(ij),iyy(1),iz2,it)
              ztsigu = targztopu(ixx(ij),iyy(1),iz2,it)
              zbsigv = targzbotv(ixx(1),iyy(ij),iz2,it)
              ztsigv = targztopv(ixx(1),iyy(ij),iz2,it)
              wgtu(iz2) = 0.0
              wgtv(iz2) = 0.0

              if (zbsigu.lt.1.e15) then
               dzsig = zbsigu-ztsigu
! compute wgt() array, which corresponds to the fraction
! of this z-layer flux that is associated with this
! isopycnal layer.  Sum over all sigma layers must give
! total horizontal flux for this z-level!
!  if this z-layer falls entirely within the isopycnal layer:
               if (zb.le.zbsigu.and.zt.ge.ztsigu) then
                wgtu(iz2) = 1.0
               end if
!  if only bottom of z-cell is within isopycnal layer:
               if (zb.ge.ztsigu.and.zt.lt.ztsigu) then
                wgtu(iz2) = (zb-ztsigu)/dz(iz)
               end if
!  if only top of z-cell is within isopycnal layer:
               if (zt.le.zbsigu.and.zb.gt.zbsigu) then
                wgtu(iz2) = (zbsigu-zt)/dz(iz)
               end if
!  if the isopycnal layer falls entirely within the z-layer:
               if (zt.lt.ztsigu.and.zb.gt.zbsigu) then
                wgtu(iz2) = dzsig/dz(iz)
               end if
              end if

              if (zbsigv.lt.1.e15) then
               dzsig = zbsigv-ztsigv
! compute wgt() array, which corresponds to the fraction
! of this z-layer flux that is associated with this
! isopycnal layer.  Sum over all sigma layers must give
! total horizontal flux for this z-level!
!  if this z-layer falls entirely within the isopycnal layer:
               if (zb.le.zbsigv.and.zt.ge.ztsigv) then
                wgtv(iz2) = 1.0
               end if
!  if only bottom of z-cell is within isopycnal layer:
               if (zb.ge.ztsigv.and.zt.lt.ztsigv) then
                wgtv(iz2) = (zb-ztsigv)/dz(iz)
               end if
!  if only top of z-cell is within isopycnal layer:
               if (zt.le.zbsigv.and.zb.gt.zbsigv) then
                wgtv(iz2) = (zbsigv-zt)/dz(iz)
               end if
!  if the isopycnal layer falls entirely within the z-layer:
               if (zt.lt.ztsigv.and.zb.gt.zbsigv) then
                wgtv(iz2) = dzsig/dz(iz)
               end if
              end if

             end do

! Use normalized wgt() array to compute tmpu, tmpv
             sumwgtu = sum(wgtu)
             sumwgtv = sum(wgtv)
             if (sumwgtu.ne.0.0) then
               do iz2=1,targnz
                if (wgtu(iz2).ne.0.) then
                 wgtu(iz2) = wgtu(iz2)/sumwgtu
                 if (tmpu(ij,iz2).eq.mval) then
                  tmpu(ij,iz2)= wgtu(iz2)*udydz
                 else
                  tmpu(ij,iz2)=tmpu(ij,iz2)+wgtu(iz2)*udydz
                 end if
                end if
               end do
             end if
             if (sumwgtv.ne.0.0) then
               do iz2=1,targnz
                if (wgtv(iz2).ne.0.) then
                 wgtv(iz2) = wgtv(iz2)/sumwgtv
                 if (tmpv(ij,iz2).eq.mval) then
                  tmpv(ij,iz2)= wgtv(iz2)*vdxdz
                 else
                  tmpv(ij,iz2)=tmpv(ij,iz2)+wgtv(iz2)*vdxdz
                 end if
                end if
               end do
             end if
            end do
           end do
!  This ends the loops over all z and all relevant U,V points.
!  Now, tmpu, tmpv are defined, so compute
!  convergence and fill vout and wout:
          do iz2=1,targnz
           uout(ix,iy,iz2,it) = tmpu(1,iz2)
           vout(ix,iy,iz2,it) = tmpv(1,iz2)
           if (all(tmpv(:,iz2).eq.mval).and. &
               all(tmpu(:,iz2).eq.mval)) then
            wout(ix,iy,iz2,it) = 0.0
           else
            where(tmpu(:,iz2).eq.mval) tmpu(:,iz2)=0.
            where(tmpv(:,iz2).eq.mval) tmpv(:,iz2)=0.
            fueE = tmpu(1,iz2)
            fueW = tmpu(2,iz2)
            fvnN = tmpv(1,iz2)
            fvnS = tmpv(2,iz2)
            wout(ix,iy,iz2,it) = -(fueE-fueW+fvnN-fvnS)
           end if
          end do

          !  vertical sum
          do iz2=1,targnz-1
            wout(ix,iy,targnz-iz2,it) = wout(ix,iy,targnz-iz2,it) + &
                 wout(ix,iy,targnz-iz2+1,it)
          end do

        end if
      end do
      end do
      end do

end subroutine sgsfluxconv
!*******************************************************************

!*******************************************************************
subroutine sig2z(nt,nz,ny,nx,nsig,popsig,kmu,z_t,zbot, &
            slevst,slevsb,zoutt,zoutb,mval)
      implicit none
      integer, intent(in) :: nt,nz,ny,nx,nsig,kmu(nx,ny)
      real, intent(in) :: mval
      real, intent(in) :: popsig(nx,ny,nz,nt)
      real, intent(in) :: slevst(nsig),slevsb(nsig)
      real, intent(in) :: z_t(nz),zbot(nz)
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
           hu = zbot(kmax)
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
