// remove this after dependencies has been added
#![allow(dead_code)]

/// # Arguments
/// * `m` `1 <= m`
///
/// # Returns
/// x mod m
/* const */
pub(crate) fn safe_mod(x: i64, m: i64) -> i64 {
    // rem_euclid: Rust 1.38.0 or later
    x.rem_euclid(m)
}

/// Fast modular by barrett reduction
/// Reference: https://en.wikipedia.org/wiki/Barrett_reduction
/// NOTE: reconsider after Ice Lake
pub(crate) struct Barrett {
    pub(crate) _m: u32,
    pub(crate) im: u64,
}

impl Barrett {
    /// # Arguments
    /// * `m` `1 <= m`
    /// (Note: `m <= 2^31` should also hold, which is undocumented in the original library.
    /// See the [pull reqeust commment](https://github.com/rust-lang-ja/ac-library-rs/pull/3#discussion_r484661007)
    /// for more details.)
    pub(crate) fn new(m: u32) -> Barrett {
        Barrett {
            _m: m,
            im: (-1i64 as u64 / m as u64).wrapping_add(1),
        }
    }

    /// # Returns
    /// `m`
    pub(crate) fn umod(&self) -> u32 {
        self._m
    }

    /// # Parameters
    /// * `a` `0 <= a < m`
    /// * `b` `0 <= b < m`
    ///
    /// # Returns
    /// a * b % m
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn mul(&self, a: u32, b: u32) -> u32 {
        mul_mod(a, b, self._m, self.im)
    }
}

/// Calculates `a * b % m`.
///
/// * `a` `0 <= a < m`
/// * `b` `0 <= b < m`
/// * `m` `1 <= m < 2^32`
/// * `im` = ceil(2^64 / `m`) = floor((2^64 - 1) / `m`) + 1
#[allow(clippy::many_single_char_names)]
pub(crate) fn mul_mod(a: u32, b: u32, m: u32, im: u64) -> u32 {
    // [1] m = 1
    // a = b = im = 0, so okay

    // [2] m >= 2
    // im = ceil(2^64 / m) = floor((2^64 - 1) / m) + 1
    // -> im * m = 2^64 + r (0 <= r < m)
    // let z = a*b = c*m + d (0 <= c, d < m)
    // a*b * im = (c*m + d) * im = c*(im*m) + d*im = c*2^64 + c*r + d*im
    // c*r + d*im < m * m + m * im < m * m + 2^64 + m <= 2^64 + m * (m + 1) < 2^64 * 2
    // ((ab * im) >> 64) == c or c + 1
    let z = (a as u64) * (b as u64);
    let x = (((z as u128) * (im as u128)) >> 64) as u64;
    match z.overflowing_sub(x.wrapping_mul(m as u64)) {
        (v, true) => (v as u32).wrapping_add(m),
        (v, false) => v as u32,
    }
}

/// # Parameters
/// * `x`
/// * `n` `0 <= n`
/// * `m` `1 <= m`
/// * `im` = ceil(2^64 / `m`) = floor((2^64 - 1) / `m`) + 1
///
/// # Returns
/// `(x ** n) % m`
/* const */
#[allow(clippy::many_single_char_names)]
pub(crate) fn pow_mod(x: u32, mut n: u32, m: u32, im: Option<u64>) -> u32 {
    if m <= 1 {
        return 0;
    }
    let im = match im {
        Some(im) => im,
        None => (-1i64 as u64 / m as u64).wrapping_add(1),
    };
    let mut r: u32 = 1;
    let mut y: u32 = match x.overflowing_sub(m) {
        (_, true) => x,
        (v, false) => v % m,
    };
    while n != 0 {
        if n % 2 != 0 {
            r = mul_mod(r, y, m, im);
        }
        y = mul_mod(y, y, m, im);
        n >>= 1;
    }
    r
}

/// # Parameters
/// * `x`
/// * `n` `0 <= n`
/// * `m` `1 <= m`
/// * `im` = ceil(2^64 / `m`) = floor((2^64 - 1) / `m`) + 1
///
/// # Returns
/// `(x ** n) % m`
/* const */
#[allow(clippy::many_single_char_names)]
pub(crate) fn pow_mod_u64(x: u64, mut n: u64, m: u32, im: Option<u64>) -> u32 {
    if m <= 1 {
        return 0;
    }
    let im = match im {
        Some(im) => im,
        None => (-1i64 as u64 / m as u64).wrapping_add(1),
    };
    let mut r: u32 = 1;
    let mut y: u32 = match x.overflowing_sub(m as u64) {
        (_, true) => x as u32,
        (v, false) => (v % m as u64) as u32,
    };
    while n != 0 {
        if n % 2 != 0 {
            r = mul_mod(r, y, m, im);
        }
        y = mul_mod(y, y, m, im);
        n >>= 1;
    }
    r
}

/// # Parameters
/// * `x`
/// * `n` `0 <= n`
/// * `m` `1 <= m`
/// * `im` = ceil(2^64 / `m`) = floor((2^64 - 1) / `m`) + 1
///
/// # Returns
/// `(x ** n) % m`
/* const */
#[allow(clippy::many_single_char_names)]
pub(crate) fn pow_mod_i64(x: i64, n: u64, m: u32, im: Option<u64>) -> u32 {
    if x < 0 {
        pow_mod_u64(m as u64 - (x as u64).wrapping_neg() % m as u64, n, m, im)
    } else {
        pow_mod_u64(x as _, n, m, im)
    }
}

/// Reference:
/// M. Forisek and J. Jancina,
/// Fast Primality Testing for Integers That Fit into a Machine Word
///
/// # Parameters
/// * `n` `0 <= n`
/* const */
pub(crate) fn is_prime(n: u32) -> bool {
    match n {
        _ if n <= 1 => return false,
        2 | 7 | 61 => return true,
        _ if n % 2 == 0 => return false,
        _ => {}
    }
    let im = (-1i64 as u64 / n as u64).wrapping_add(1);
    let mut d = n - 1;
    d >>= d.trailing_zeros();
    for &a in &[2, 7, 61] {
        let mut t = d;
        let mut y = pow_mod(a, t, n, Some(im));
        while t != n - 1 && y != 1 && y != n - 1 {
            y = mul_mod(y, y, n, im);
            t <<= 1;
        }
        if y != n - 1 && t % 2 == 0 {
            return false;
        }
    }
    true
}

// omitted
// template <int n> constexpr bool is_prime = is_prime_constexpr(n);

/// # Parameters
/// * `b` `1 <= b`
///
/// # Returns
/// (g, x) s.t. g = gcd(a, b), xa = g (mod b), 0 <= x <= b/g
/* const */
#[allow(clippy::many_single_char_names)]
pub(crate) fn inv_gcd<U>(a: U, b: U) -> (U, U)
where
    U: Copy
        + std::cmp::Ord
        + std::convert::From<u8>
        + std::ops::AddAssign
        + std::ops::Sub<Output = U>
        + std::ops::Mul<Output = U>
        + std::ops::Div<Output = U>
        + std::ops::RemAssign,
{
    // Constracts:
    // [1] s - m0 * a == 0 (mod b)
    // [2] t + m1 * a == 0 (mod b)
    // [3] t * m0 + s * m1 == b
    let (mut s, mut t, mut m0, mut m1) = (a, b, U::from(1), U::from(0));
    loop {
        if s == U::from(0) {
            // if a == s == 0 => (m0, m1) == (1, 0) => m0 > m1
            // if s' == 0 => v > 0 => m0' == m0 + v * m1' >= m1' => m0' >= m1'
            // s==0,[1],[2] => m0 * a mod b == 0, -m1 * a mod b == t => (m0 - m1) * a mod b == t
            // m0 == b / t => m0 - m1 <= b / t
            break (t, m0 - m1);
        }
        // u := floor(t / s)
        // m1' := m1 + u * m0
        m1 += (t / s) * m0;
        // t' := t - s * u;
        t %= s;
        // [2']: t' + m1' * a == 0 (mod b)
        // [3']:
        // t' * m0 + s * m1'
        // == (t - s * u) * m0 + s * (m1 + u * m0)
        // == t * m0 + s * m1 == b
        if t == U::from(0) {
            // t==0,[1] => m0 * a mod b == s
            // u > 0 => b / s == m1' == m1 + m0 * u >= m0
            break (s, m0);
        }
        // v := floor(s / t')
        // m0' := m0 + v * m1'
        m0 += (s / t) * m1;
        // s' := s - t' * v
        s %= t;
        // [1']: s' - m0' * a == 0 (mod b)
        // [3'']:
        // t' * m0' + s' * m1'
        // == t' * (m0 + v * m1') + (s - t' * v) * m1'
        // == t' * m0 + s * m1' == b
    }
}

/// # Parameters
/// * `b` `1 <= b`
///
/// # Returns
/// (g, x) s.t. g = gcd(a, b), xa = g (mod b), 0 <= x <= b/g
#[allow(clippy::many_single_char_names)]
pub(crate) fn iinv_gcd(a: i64, b: u64) -> (u64, u64) {
    if a < 0 {
        let (g, i) = inv_gcd((a as u64).wrapping_neg(), b);
        (g, b / g - i)
    } else {
        inv_gcd(a as u64, b)
    }
}

/// Compile time (currently not) primitive root
/// @param m must be prime
/// @return primitive root (and minimum in now)
/* const */
pub(crate) fn primitive_root(m: u32) -> u32 {
    match m {
        2 => return 1,
        167_772_161 => return 3,
        469_762_049 => return 3,
        754_974_721 => return 11,
        998_244_353 => return 3,
        _ => {}
    }
    let im = (-1i64 as u64 / m as u64).wrapping_add(1);
    let mut divs = [0; 20];
    divs[0] = 2;
    let mut cnt = 1;
    let mut x = (m - 1) / 2;
    while x % 2 == 0 {
        x /= 2;
    }
    for i in (3..u32::max_value()).step_by(2) {
        if i as u64 * i as u64 > x as u64 {
            break;
        }
        if x % i == 0 {
            divs[cnt] = i;
            cnt += 1;
            while x % i == 0 {
                x /= i;
            }
        }
    }
    if x > 1 {
        divs[cnt] = x;
        cnt += 1;
    }
    let mut g: u32 = 2;
    loop {
        if (0..cnt).all(|i| pow_mod(g, (m - 1) / divs[i], m, Some(im)) != 1) {
            break g;
        }
        g += 1;
    }
}
// omitted
// template <int m> constexpr int primitive_root = primitive_root_constexpr(m);

#[cfg(target_arch = "x86_64")]
pub(crate) fn x86_rdrand(limit: u64) -> u64 {
    let mut rand = 0;
    unsafe { std::arch::x86_64::_rdrand64_step(&mut rand) };
    ((rand as u128 * limit as u128) >> 64) as u64
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal)]
    #![allow(clippy::cognitive_complexity)]
    use crate::internal_math::{
        iinv_gcd, inv_gcd, is_prime, pow_mod, pow_mod_u64, primitive_root, safe_mod, Barrett,
    };
    use std::collections::HashSet;

    #[test]
    fn test_safe_mod() {
        assert_eq!(safe_mod(0, 3), 0);
        assert_eq!(safe_mod(1, 3), 1);
        assert_eq!(safe_mod(2, 3), 2);
        assert_eq!(safe_mod(3, 3), 0);
        assert_eq!(safe_mod(4, 3), 1);
        assert_eq!(safe_mod(5, 3), 2);
        assert_eq!(safe_mod(73, 11), 7);
        assert_eq!(safe_mod(2306249155046129918, 6620319213327), 1374210749525);

        assert_eq!(safe_mod(-1, 3), 2);
        assert_eq!(safe_mod(-2, 3), 1);
        assert_eq!(safe_mod(-3, 3), 0);
        assert_eq!(safe_mod(-4, 3), 2);
        assert_eq!(safe_mod(-5, 3), 1);
        assert_eq!(safe_mod(-7170500492396019511, 777567337), 333221848);
    }

    #[test]
    fn test_barrett() {
        let b = Barrett::new(7);
        assert_eq!(b.umod(), 7);
        assert_eq!(b.mul(2, 3), 6);
        assert_eq!(b.mul(4, 6), 3);
        assert_eq!(b.mul(5, 0), 0);

        let b = Barrett::new(998244353);
        assert_eq!(b.umod(), 998244353);
        assert_eq!(b.mul(2, 3), 6);
        assert_eq!(b.mul(3141592, 653589), 919583920);
        assert_eq!(b.mul(323846264, 338327950), 568012980);

        // make `z - x * self._m as u64` overflow.
        // Thanks @koba-e964 (at https://github.com/rust-lang-ja/ac-library-rs/pull/3#discussion_r484932161)
        let b = Barrett::new(2147483647);
        assert_eq!(b.umod(), 2147483647);
        assert_eq!(b.mul(1073741824, 2147483645), 2147483646);

        // test `2^31 < self._m < 2^32` case.
        let b = Barrett::new(3221225471);
        assert_eq!(b.umod(), 3221225471);
        assert_eq!(b.mul(3188445886, 2844002853), 1840468257);
        assert_eq!(b.mul(2834869488, 2779159607), 2084027561);
        assert_eq!(b.mul(3032263594, 3039996727), 2130247251);
        assert_eq!(b.mul(3029175553, 3140869278), 1892378237);
    }

    #[test]
    fn test_pow_mod() {
        assert_eq!(pow_mod(0, 0, 1, None), 0);
        assert_eq!(pow_mod(0, 0, 3, None), 1);
        assert_eq!(pow_mod(0, 0, 723, None), 1);
        assert_eq!(pow_mod(0, 0, 998244353, None), 1);
        assert_eq!(pow_mod(0, 0, i32::max_value() as u32, None), 1);

        assert_eq!(pow_mod(0, 1, 1, None), 0);
        assert_eq!(pow_mod(0, 1, 3, None), 0);
        assert_eq!(pow_mod(0, 1, 723, None), 0);
        assert_eq!(pow_mod(0, 1, 998244353, None), 0);
        assert_eq!(pow_mod(0, 1, i32::max_value() as u32, None), 0);

        assert_eq!(pow_mod_u64(0, i64::max_value() as u64, 1, None), 0);
        assert_eq!(pow_mod_u64(0, i64::max_value() as u64, 3, None), 0);
        assert_eq!(pow_mod_u64(0, i64::max_value() as u64, 723, None), 0);
        assert_eq!(pow_mod_u64(0, i64::max_value() as u64, 998244353, None), 0);
        assert_eq!(
            pow_mod_u64(0, i64::max_value() as u64, i32::max_value() as u32, None),
            0
        );

        assert_eq!(pow_mod(1, 0, 1, None), 0);
        assert_eq!(pow_mod(1, 0, 3, None), 1);
        assert_eq!(pow_mod(1, 0, 723, None), 1);
        assert_eq!(pow_mod(1, 0, 998244353, None), 1);
        assert_eq!(pow_mod(1, 0, i32::max_value() as u32, None), 1);

        assert_eq!(pow_mod(1, 1, 1, None), 0);
        assert_eq!(pow_mod(1, 1, 3, None), 1);
        assert_eq!(pow_mod(1, 1, 723, None), 1);
        assert_eq!(pow_mod(1, 1, 998244353, None), 1);
        assert_eq!(pow_mod(1, 1, i32::max_value() as u32, None), 1);

        assert_eq!(pow_mod_u64(1, i64::max_value() as u64, 1, None), 0);
        assert_eq!(pow_mod_u64(1, i64::max_value() as u64, 3, None), 1);
        assert_eq!(pow_mod_u64(1, i64::max_value() as u64, 723, None), 1);
        assert_eq!(pow_mod_u64(1, i64::max_value() as u64, 998244353, None), 1);
        assert_eq!(
            pow_mod_u64(1, i64::max_value() as u64, i32::max_value() as u32, None),
            1
        );

        assert_eq!(pow_mod_u64(i64::max_value() as u64, 0, 1, None), 0);
        assert_eq!(pow_mod_u64(i64::max_value() as u64, 0, 3, None), 1);
        assert_eq!(pow_mod_u64(i64::max_value() as u64, 0, 723, None), 1);
        assert_eq!(pow_mod_u64(i64::max_value() as u64, 0, 998244353, None), 1);
        assert_eq!(
            pow_mod_u64(i64::max_value() as u64, 0, i32::max_value() as u32, None),
            1
        );

        assert_eq!(
            pow_mod_u64(i64::max_value() as u64, i64::max_value() as u64, 1, None),
            0
        );
        assert_eq!(
            pow_mod_u64(i64::max_value() as u64, i64::max_value() as u64, 3, None),
            1
        );
        assert_eq!(
            pow_mod_u64(i64::max_value() as u64, i64::max_value() as u64, 723, None),
            640
        );
        assert_eq!(
            pow_mod_u64(
                i64::max_value() as u64,
                i64::max_value() as u64,
                998244353,
                None
            ),
            683296792
        );
        assert_eq!(
            pow_mod_u64(
                i64::max_value() as u64,
                i64::max_value() as u64,
                i32::max_value() as u32,
                None
            ),
            1
        );

        assert_eq!(pow_mod(2, 3, 1_000_000_007, None), 8);
        assert_eq!(pow_mod(5, 7, 1_000_000_007, None), 78125);
        assert_eq!(pow_mod(123, 456, 1_000_000_007, None), 565291922);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(6));
        assert!(is_prime(7));
        assert!(!is_prime(8));
        assert!(!is_prime(9));

        // assert!(is_prime(57));
        assert!(!is_prime(57));
        assert!(!is_prime(58));
        assert!(is_prime(59));
        assert!(!is_prime(60));
        assert!(is_prime(61));
        assert!(!is_prime(62));

        assert!(!is_prime(701928443));
        assert!(is_prime(998244353));
        assert!(!is_prime(1_000_000_000));
        assert!(is_prime(1_000_000_007));

        assert!(is_prime(i32::max_value() as u32));
    }

    #[test]
    fn test_is_prime_sieve() {
        let n = 1_000_000;
        let mut prime = vec![true; n];
        prime[0] = false;
        prime[1] = false;
        for i in 0..n {
            assert_eq!(prime[i], is_prime(i as u32));
            if prime[i] {
                for j in (2 * i..n).step_by(i) {
                    prime[j] = false;
                }
            }
        }
    }

    #[test]
    fn test_inv_gcd() {
        for &(a, b, g) in &[
            (0, 1, 1),
            (0, 4, 4),
            (0, 7, 7),
            (2, 3, 1),
            (4, 6, 2),
            (13, 23, 1),
            (57, 81, 3),
            (12345, 67890, 15),
            (
                i64::max_value() as u64,
                i64::max_value() as u64,
                i64::max_value() as u64,
            ),
            (u64::max_value(), u64::max_value(), u64::max_value()),
        ] {
            let (g_, x) = inv_gcd(a, b);
            assert_eq!(g, g_);
            let b_ = b as i128;
            assert_eq!(((x as i128 * a as i128) % b_ + b_) % b_, g as i128 % b_);
        }
    }

    #[test]
    fn test_iinv_gcd() {
        for &(a, b, g) in &[
            (0, 1, 1),
            (0, 4, 4),
            (0, 7, 7),
            (2, 3, 1),
            (-2, 3, 1),
            (4, 6, 2),
            (-4, 6, 2),
            (13, 23, 1),
            (57, 81, 3),
            (12345, 67890, 15),
            (-3141592 * 6535, 3141592 * 8979, 3141592),
            (
                i64::max_value(),
                i64::max_value() as u64,
                i64::max_value() as u64,
            ),
            (i64::min_value(), i64::max_value() as u64, 1),
        ] {
            let (g_, x) = iinv_gcd(a, b);
            assert_eq!(g, g_);
            let b_ = b as i128;
            assert_eq!(((x as i128 * a as i128) % b_ + b_) % b_, g as i128 % b_);
        }
    }

    #[test]
    fn test_primitive_root() {
        for &p in &[
            2,
            3,
            5,
            7,
            233,
            200003,
            167_772_161,   // 0xa000001
            469_762_049,   // 0x1c000001
            754_974_721,   // 0x2d000001
            998_244_353,   // 0x3b800001
            1_000_000_007, // 0x3b9aca07
            1_811_939_329, // 0x6c000001
            2_013_265_921, // 0x78000001
            2_113_929_217, // 0x7e000001
            2_147_483_647, // 0x7fffffff
            2_717_908_993, // 0xa2000001
            3_221_225_473, // 0xc0000001
            3_489_660_929, // 0xd0000001
            3_892_314_113, // 0xe8000001
            4_076_863_489, // 0xf3000001
            4_194_304_001, // 0xfa000001
            4_294_967_291, // 0xfffffffb
        ] {
            assert!(is_prime(p));
            let g = primitive_root(p);
            if p != 2 {
                assert_ne!(g, 1);
            }

            let q = p - 1;
            for i in (2..u32::max_value()).take_while(|i| i * i <= q) {
                if q % i != 0 {
                    break;
                }
                for &r in &[i, q / i] {
                    assert_ne!(pow_mod(g, r, p, None), 1);
                }
            }
            assert_eq!(pow_mod(g, q, p, None), 1);

            if p < 1_000_000 {
                assert_eq!(
                    (0..p - 1)
                        .scan(1, |i, _| {
                            *i = *i * g % p;
                            Some(*i)
                        })
                        .collect::<HashSet<_>>()
                        .len() as u32,
                    p - 1
                );
            }
        }
    }
}
