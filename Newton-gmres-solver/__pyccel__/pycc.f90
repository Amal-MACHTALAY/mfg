module pycc

  use, intrinsic :: ISO_C_BINDING

  implicit none

  contains

  !........................................
  pure subroutine identity(n, I) 

    implicit none

    integer(C_INT64_T), value :: n
    real(C_DOUBLE), intent(inout) :: I(0:,0:)
    integer(C_INT64_T) :: i

    do i = 0_C_INT64_T, n - 1_C_INT64_T, 1_C_INT64_T
      I(i, i) = 1.0_C_DOUBLE
    end do

  end subroutine identity
  !........................................

end module pycc
