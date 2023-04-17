use bencher::{benchmark_group, benchmark_main, Bencher};
use rand::Rng;

const N: usize = 100_000;

macro_rules! staticmodint_tests {
    ($modint:ident, $addfn:ident, $subfn:ident, $mulfn:ident, $divfn:ident) => {
        fn $addfn(bench: &mut Bencher) {
            use ac_library::$modint;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $modint::modulus()))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $modint::raw(0);
                for &e in values.iter() {
                    val += $modint::raw(e);
                }
                val.val()
            })
        }
        fn $subfn(bench: &mut Bencher) {
            use ac_library::$modint;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $modint::modulus()))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $modint::raw(0);
                for &e in values.iter() {
                    val -= $modint::raw(e);
                }
                val.val()
            })
        }
        fn $mulfn(bench: &mut Bencher) {
            use ac_library::$modint;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $modint::modulus()))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $modint::raw(1);
                for &e in values.iter() {
                    val *= $modint::raw(e);
                }
                val.val()
            })
        }
        fn $divfn(bench: &mut Bencher) {
            use ac_library::$modint;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $modint::modulus()))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $modint::raw(1);
                for &e in values.iter() {
                    val /= $modint::raw(e);
                }
                val.val()
            })
        }
    };
}

macro_rules! dynamicmodint_tests {
    ($modulus:literal, $addfn:ident, $subfn:ident, $mulfn:ident, $divfn:ident) => {
        fn $addfn(bench: &mut Bencher) {
            use ac_library::ModInt;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $modulus))
                .collect::<Vec<u32>>();
            ModInt::set_modulus($modulus);
            bench.iter(|| -> u32 {
                let mut val = ModInt::raw(0);
                for &e in values.iter() {
                    val += ModInt::raw(e);
                }
                val.val()
            })
        }
        fn $subfn(bench: &mut Bencher) {
            use ac_library::ModInt;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $modulus))
                .collect::<Vec<u32>>();
            ModInt::set_modulus($modulus);
            bench.iter(|| -> u32 {
                let mut val = ModInt::raw(0);
                for &e in values.iter() {
                    val -= ModInt::raw(e);
                }
                val.val()
            })
        }
        fn $mulfn(bench: &mut Bencher) {
            use ac_library::ModInt;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $modulus))
                .collect::<Vec<u32>>();
            ModInt::set_modulus($modulus);
            bench.iter(|| -> u32 {
                let mut val = ModInt::raw(1);
                for &e in values.iter() {
                    val *= ModInt::raw(e);
                }
                val.val()
            })
        }
        fn $divfn(bench: &mut Bencher) {
            use ac_library::ModInt;
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $modulus))
                .collect::<Vec<u32>>();
            ModInt::set_modulus($modulus);
            bench.iter(|| -> u32 {
                let mut val = ModInt::raw(1);
                for &e in values.iter() {
                    val /= ModInt::raw(e);
                }
                val.val()
            })
        }
    };
}

pub trait HandModInt {
    const MOD: u32;
    const IM: u64;
    const IBIN: [u32; 64];
    fn add_impl(lhs: u32, rhs: u32) -> u32;
    fn sub_impl(lhs: u32, rhs: u32) -> u32;
    fn mul_impl(lhs: u32, rhs: u32) -> u32;
    fn div_impl(lhs: u32, rhs: u32) -> u32;
}
macro_rules! handmodint_impl {
    ($modulus:literal, $st:ident) => {
        handmodint_impl!(@dol ($) $modulus, $st);
    };
    (@dol ($dol:tt) $modulus:literal, $st:ident) => {
        pub struct $st(u32);
        impl HandModInt for $st {
            const MOD: u32 = $modulus;
            const IM: u64 = ((!0u64) / $modulus).wrapping_add(1);
            const IBIN: [u32; 64] = {
                let mut ibin = [0u32; 64];
                let m = ($modulus as u32) >> ($modulus as u32).trailing_zeros();
                ibin[0] = 1;
                const fn div2(x: u32, m: u32) -> u32 {
                    x / 2 + ((x & 1).wrapping_neg() & ((m + 1) / 2))
                }
                macro_rules! div2 {
                    ($dol($dol i:literal),*) => {
                        $dol(ibin[$dol i + 1] = div2(ibin[$dol i], m);)*
                    };
                }
                div2!(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62);
                ibin
            };
            #[inline]
            fn add_impl(lhs: u32, rhs: u32) -> u32 {
                let v = u64::from(lhs) + u64::from(rhs);
                match v.overflowing_sub(u64::from(Self::MOD)) {
                    (_, true) => v as u32,
                    (w, false) => w as u32,
                }
            }
            #[inline]
            fn sub_impl(lhs: u32, rhs: u32) -> u32 {
                match lhs.overflowing_sub(rhs) {
                    (v, true) => v.wrapping_add(Self::MOD),
                    (v, false) => v,
                }
            }
            #[inline]
            fn mul_impl(lhs: u32, rhs: u32) -> u32 {
                /*
                let z = (lhs as u64) * (rhs as u64);
                let x = (((z as u128) * (Self::IM as u128)) >> 64) as u64;
                match z.overflowing_sub(x.wrapping_mul(Self::MOD as u64)) {
                    (v, true) => (v as u32).wrapping_add(Self::MOD),
                    (v, false) => v as u32,
                }
                */
                (u64::from(lhs) * u64::from(rhs) % u64::from(Self::MOD)) as u32
            }
            #[inline]
            fn div_impl(lhs: u32, rhs: u32) -> u32 {
                assert_ne!(rhs, 0, "{} / {} mod {}; divide zero", lhs, rhs, Self::MOD);
                assert!(Self::MOD % 2 != 0 || rhs % 2 != 0, "{} / {} mod {}; gcd({}, {}) >= 2; modular inverse does not exist.", lhs, rhs, Self::MOD, rhs, Self::MOD);
                let (mut x, mut y, mut a, mut b) = (1, 0, rhs, Self::MOD);
                let mut s = a.trailing_zeros();
                a >>= s;
                let t = b.trailing_zeros();
                x <<= t;
                b >>= t;
                s += t;
                loop {
                    if a == 1 {
                        return Self::mul_impl(lhs, Self::mul_impl(x, Self::IBIN[s as usize]));
                    }
                    assert_ne!(a, b, "{} / {} mod {}; gcd({}, {}) = {}; modular inverse does not exist.", lhs, rhs, Self::MOD, rhs, Self::MOD, a);
                    if a < (b >> 3) {
                        y += x * (b / a);
                        b %= a;
                        assert_ne!(b, 0, "{} / {} mod {}; gcd({}, {}) = {}; modular inverse does not exist.", lhs, rhs, Self::MOD, rhs, Self::MOD, b);
                        let t = b.trailing_zeros();
                        x <<= t;
                        b >>= t;
                        s += t;
                    } else {
                        while a < b {
                            y += x;
                            b -= a;
                            let t = b.trailing_zeros();
                            x <<= t;
                            b >>= t;
                            s += t;
                        }
                    }
                    if b == 1 {
                        return Self::mul_impl(
                            lhs,
                            Self::mul_impl(Self::sub_impl(0, y), Self::IBIN[s as usize]),
                        );
                    }
                    if (a >> 3) > b {
                        x += y * (a / b);
                        a %= b;
                        assert_ne!(a, 0, "{} / {} mod {}; gcd({}, {}) = {}; modular inverse does not exist.", lhs, rhs, Self::MOD, rhs, Self::MOD, a);
                        let t = a.trailing_zeros();
                        y <<= t;
                        a >>= t;
                        s += t;
                    } else {
                        while a > b {
                            x += y;
                            a -= b;
                            let t = a.trailing_zeros();
                            y <<= t;
                            a >>= t;
                            s += t;
                        }
                    }
                }
            }
        }
        impl std::ops::Add for $st {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                Self(Self::add_impl(self.0, rhs.0))
            }
        }
        impl std::ops::Sub for $st {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self(Self::sub_impl(self.0, rhs.0))
            }
        }
        impl std::ops::Mul for $st {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                Self(Self::mul_impl(self.0, rhs.0))
            }
        }
        impl std::ops::Div for $st {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                Self(Self::div_impl(self.0, rhs.0))
            }
        }
        impl std::ops::Neg for $st {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self(Self::sub_impl(0, self.0))
            }
        }
        impl std::ops::AddAssign for $st {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = Self(Self::add_impl(self.0, rhs.0));
            }
        }
        impl std::ops::SubAssign for $st {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = Self(Self::sub_impl(self.0, rhs.0));
            }
        }
        impl std::ops::MulAssign for $st {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = Self(Self::mul_impl(self.0, rhs.0));
            }
        }
        impl std::ops::DivAssign for $st {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = Self(Self::div_impl(self.0, rhs.0));
            }
        }
    };
}

handmodint_impl!(167_772_161, HandModInt167772161);
handmodint_impl!(469_762_049, HandModInt469762049);
handmodint_impl!(754_974_721, HandModInt754974721);
handmodint_impl!(998_244_353, HandModInt998244353);
handmodint_impl!(1_000_000_007, HandModInt1000000007);
handmodint_impl!(4_076_863_489, HandModInt4076863489);

macro_rules! handmodint_tests {
    ($st:ident, $addfn:ident, $subfn:ident, $mulfn:ident, $divfn:ident) => {
        fn $addfn(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $st::MOD))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $st(0);
                for &e in values.iter() {
                    val += $st(e);
                }
                val.0
            })
        }
        fn $subfn(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(0, $st::MOD))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $st(0);
                for &e in values.iter() {
                    val -= $st(e);
                }
                val.0
            })
        }
        fn $mulfn(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $st::MOD))
                .collect::<Vec<u32>>();
            bench.iter(|| -> u32 {
                let mut val = $st(1);
                for &e in values.iter() {
                    val *= $st(e);
                }
                val.0
            })
        }
        fn $divfn(bench: &mut Bencher) {
            let mut rng = rand::thread_rng();
            let values = (0..N)
                .map(|_| rng.gen_range(1, $st::MOD))
                .collect::<Vec<u32>>();
            use ac_library::ModInt;
            ModInt::set_modulus($st::MOD);
            let v0 = values
                .iter()
                .fold(ModInt::raw(1), |p, &c| p / ModInt::raw(c))
                .val();
            let v1 = values.iter().fold($st(1), |p, &c| p / $st(c)).0;
            assert_eq!(v0, v1);
            bench.iter(|| -> u32 {
                let mut val = $st(1);
                for &e in values.iter() {
                    val /= $st(e);
                }
                val.0
            })
        }
    };
}

staticmodint_tests!(
    ModInt998244353,
    smod0998244353_add,
    smod0998244353_sub,
    smod0998244353_mul,
    smod0998244353_div
);
staticmodint_tests!(
    ModInt1000000007,
    smod1000000007_add,
    smod1000000007_sub,
    smod1000000007_mul,
    smod1000000007_div
);
dynamicmodint_tests!(
    167_772_161,
    dmod0167772161_add,
    dmod0167772161_sub,
    dmod0167772161_mul,
    dmod0167772161_div
);
dynamicmodint_tests!(
    469_762_049,
    dmod0469762049_add,
    dmod0469762049_sub,
    dmod0469762049_mul,
    dmod0469762049_div
);
dynamicmodint_tests!(
    4_076_863_489,
    dmod4076863489_add,
    dmod4076863489_sub,
    dmod4076863489_mul,
    dmod4076863489_div
);
dynamicmodint_tests!(
    754_974_721,
    dmod0754974721_add,
    dmod0754974721_sub,
    dmod0754974721_mul,
    dmod0754974721_div
);
handmodint_tests!(
    HandModInt167772161,
    hmod0167772161_add,
    hmod0167772161_sub,
    hmod0167772161_mul,
    hmod0167772161_div
);
handmodint_tests!(
    HandModInt469762049,
    hmod0469762049_add,
    hmod0469762049_sub,
    hmod0469762049_mul,
    hmod0469762049_div
);
handmodint_tests!(
    HandModInt754974721,
    hmod0754974721_add,
    hmod0754974721_sub,
    hmod0754974721_mul,
    hmod0754974721_div
);
handmodint_tests!(
    HandModInt998244353,
    hmod0998244353_add,
    hmod0998244353_sub,
    hmod0998244353_mul,
    hmod0998244353_div
);
handmodint_tests!(
    HandModInt1000000007,
    hmod1000000007_add,
    hmod1000000007_sub,
    hmod1000000007_mul,
    hmod1000000007_div
);
handmodint_tests!(
    HandModInt4076863489,
    hmod4076863489_add,
    hmod4076863489_sub,
    hmod4076863489_mul,
    hmod4076863489_div
);

benchmark_group!(
    benches,
    dmod0167772161_add,
    dmod0167772161_sub,
    dmod0167772161_mul,
    dmod0167772161_div,
    dmod0469762049_add,
    dmod0469762049_sub,
    dmod0469762049_mul,
    dmod0469762049_div,
    dmod0754974721_add,
    dmod0754974721_sub,
    dmod0754974721_mul,
    dmod0754974721_div,
    dmod4076863489_add,
    dmod4076863489_sub,
    dmod4076863489_mul,
    dmod4076863489_div,
    hmod0167772161_add,
    hmod0167772161_sub,
    hmod0167772161_mul,
    hmod0167772161_div,
    hmod0469762049_add,
    hmod0469762049_sub,
    hmod0469762049_mul,
    hmod0469762049_div,
    hmod0754974721_add,
    hmod0754974721_sub,
    hmod0754974721_mul,
    hmod0754974721_div,
    hmod0998244353_add,
    hmod0998244353_sub,
    hmod0998244353_mul,
    hmod0998244353_div,
    hmod1000000007_add,
    hmod1000000007_sub,
    hmod1000000007_mul,
    hmod1000000007_div,
    hmod4076863489_add,
    hmod4076863489_sub,
    hmod4076863489_mul,
    hmod4076863489_div,
    smod0998244353_add,
    smod0998244353_sub,
    smod0998244353_mul,
    smod0998244353_div,
    smod1000000007_add,
    smod1000000007_sub,
    smod1000000007_mul,
    smod1000000007_div,
);
benchmark_main!(benches);
