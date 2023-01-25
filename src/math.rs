//! Number-theoretic algorithms.

use crate::internal_math;

use std::mem::swap;

/// Returns $x^n \bmod m$.
///
/// # Constraints
///
/// - $0 \leq n$
/// - $1 \leq m$
///
/// # Panics
///
/// Panics if the above constraints are not satisfied.
///
/// # Complexity
///
/// - $O(\log n)$
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// assert_eq!(math::pow_mod(2, 10000, 7), 2);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn pow_mod(x: u64, mut n: u64, m: u32) -> u32 {
    assert!(1 <= m && m <= 2u32.pow(31));
    if m == 1 {
        return 0;
    }
    let bt = internal_math::Barrett::new(m);
    let mut r = 1;
    let mut y = (x % m as u64) as u32;
    while n != 0 {
        if n & 1 != 0 {
            r = bt.mul(r, y);
        }
        y = bt.mul(y, y);
        n >>= 1;
    }
    r
}

/// Returns $x^n \bmod m$.
///
/// # Constraints
///
/// - $0 \leq n$
/// - $1 \leq m$
///
/// # Panics
///
/// Panics if the above constraints are not satisfied.
///
/// # Complexity
///
/// - $O(\log n)$
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// assert_eq!(math::pow_mod(2, 10000, 7), 2);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn ipow_mod(x: i64, mut n: u64, m: u32) -> u32 {
    assert!(1 <= m && m <= 2u32.pow(31));
    if m == 1 {
        return 0;
    }
    let bt = internal_math::Barrett::new(m);
    let mut r = 1;
    let mut y = x.rem_euclid(m as i64) as u32;
    while n != 0 {
        if n & 1 != 0 {
            r = bt.mul(r, y);
        }
        y = bt.mul(y, y);
        n >>= 1;
    }
    r
}

/// Returns an integer $y \in [0, m)$ such that $xy \equiv 1 \pmod m$.
///
/// # Constraints
///
/// - $\gcd(x, m) = 1$
/// - $1 \leq m$
///
/// # Panics
///
/// Panics if the above constraints are not satisfied.
///
/// # Complexity
///
/// - $O(\log m)$
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// assert_eq!(math::inv_mod::<u8>(3, 7), 5);
/// assert_eq!(math::inv_mod::<u16>(3, 7), 5);
/// assert_eq!(math::inv_mod::<u32>(3, 7), 5);
/// assert_eq!(math::inv_mod::<u64>(3, 7), 5);
/// assert_eq!(math::inv_mod::<u128>(3, 7), 5);
/// ```
pub fn inv_mod<U>(x: U, m: U) -> U
where
    U: Copy
        + std::cmp::Ord
        + std::convert::From<u8>
        + std::convert::Into<u128>
        + std::ops::AddAssign
        + std::ops::Sub<Output = U>
        + std::ops::Mul<Output = U>
        + std::ops::Div<Output = U>
        + std::ops::RemAssign,
{
    assert!(U::from(1) <= m);
    let z = internal_math::inv_gcd(x, m);
    assert!(z.0 == U::from(1));
    z.1
}

/// Returns integer $\lfloor\sqrt{a}\rfloor$.
///
/// # Complexity
///
/// - $O(\log\log a)$, when add/sub/mul/div/rem are considered as constant time
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// assert_eq!(math::sqrt_floor::<u8>(255), 15);
/// assert_eq!(math::sqrt_floor::<u16>(65_026), 255);
/// assert_eq!(math::sqrt_floor::<u32>(4_294_836_226), 65_535);
/// assert_eq!(math::sqrt_floor::<u64>(18_446_744_065_119_617_027), 4_294_967_295);
/// assert_eq!(math::sqrt_floor::<u128>(340_282_366_920_938_463_426_481_119_284_349_108_226), 18_446_744_073_709_551_615);
/// ```
///
/// # Algorithm
///
/// Integer Square-Root with Floor function using Newton-Raphson method
///
/// - $s_n\in\mathbb{N},\ a\in\mathbb{N},\ s_n\ge 1,\ a\ge 1, \mathbb{N}\text{ is the set of natural numbers}$
/// - $s_0\ge\lfloor\sqrt{a}\rfloor,\quad\lfloor\sqrt{a}\rfloor\le 2^{\lceil\log_2(\sqrt{a})\rceil}\lt 2\sqrt{a}$
/// - $s_{n+1}=\lfloor(s_n+\lfloor a/s_n\rfloor)/2\rfloor$
/// - $a\le 2^{4},\quad n\ge 1,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
/// - $a\le 2^{10},\quad n\ge 2,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
/// - $a\le 2^{23},\quad n\ge 3,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
/// - $a\le 2^{48},\quad n\ge 4,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
/// - $a\le 2^{99},\quad n\ge 5,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
/// - $a\le 2^{200},\quad n\ge 6,\quad \lfloor\sqrt{a}\rfloor\le s_0\le 2\sqrt{a} \quad\Rightarrow\quad \lfloor\sqrt{a}\rfloor\le s_n\le\lfloor\sqrt{a}\rfloor+1$
///
/// # Proof
///
/// - (lemma) Since $s_n$ is a natural number,
///   $$
///   s_{n+1}=\left\lfloor\left(s_n+\left\lfloor\frac{a}{s_n}\right\rfloor\right)/2\right\rfloor=\left\lfloor\left\lfloor s_n+\frac{a}{s_n}\right\rfloor/2\right\rfloor=\left\lfloor\left(s_n+\frac{a}{s_n}\right)/2\right\rfloor=\left\lfloor\frac{s_n^2+a}{2s_n}\right\rfloor
///   $$
/// - (1: Monotonically decreasing when not reached) case $\ s_n\gt\left\lfloor\sqrt{a}\right\rfloor$,
///   - Since $s_n$ is a natural number, $\ s_n\gt\left\lfloor\sqrt{a}\right\rfloor\ \Rightarrow\ s_n\gt\sqrt{a}$
///   - $\ s_n=(1+\varepsilon)\sqrt{a}\ $ with $\ \varepsilon\gt 0\ $ then
///     $$
///     \left\lfloor\sqrt{a}\right\rfloor\le\left\lfloor\left(1+\frac{\varepsilon^2}{2(1+\varepsilon)}\right)\sqrt{a}\right\rfloor=\left\lfloor\frac{s_n^2+a}{2s_n}\right\rfloor=s_{n+1}\le\frac{s_n^2+a}{2s_n}\lt\frac{s_n^2+s_n^2}{2s_n}=s_n
///     $$
///   - $\therefore\quad s_n\gt\left\lfloor\sqrt{a}\right\rfloor\ \Rightarrow\ \left\lfloor\sqrt{a}\right\rfloor\le s_{n+1}\lt s_n$
/// - (2: Behavior when $\left\lfloor\sqrt{a}\right\rfloor$ reaching) case $\ s_n=\left\lfloor\sqrt{a}\right\rfloor\$,
///   - $s_n=\left\lfloor\sqrt{a}\right\rfloor\ \Rightarrow\ s_n\le\sqrt{a}\lt s_n+1\ \Rightarrow\ s_n^2\le a\lt(s_n+1)^2$
///   - Since $s_n\ge 1$ is a natural number, $\ \lfloor s_n\rfloor=s_n\ $ and $\ \displaystyle{\frac{1}{2s_n}\lt 1},\ $ then
///     $$
///     \left\lfloor\sqrt{a}\right\rfloor=\left\lfloor\frac{s_n^2+s_n^2}{2s_n}\right\rfloor\le s_{n+1}\le\left\lfloor\frac{s_n^2+(s_n+1)^2}{2s_n}\right\rfloor=\left\lfloor s_n+1+\frac{1}{2s_n}\right\rfloor=\left\lfloor\sqrt{a}\right\rfloor+1
///     $$
///   - $\therefore\quad s_n=\left\lfloor\sqrt{a}\right\rfloor\ \Rightarrow\ \left\lfloor\sqrt{a}\right\rfloor\le s_{n+1}\le\left\lfloor\sqrt{a}\right\rfloor+1$
/// - (3: Convergence of the Newton-Raphson method)
///   $$
///   \varepsilon\gt 0,\quad s_n=(1+\varepsilon)\sqrt{a},\quad s_{n+1}=\left\lfloor\left(1+\frac{\varepsilon^2}{2(1+\varepsilon)}\right)\sqrt{a}\right\rfloor
///   $$
///
///   Removing the floor function does not speed up convergence, so consider removing it,
///
///   $$
///   (1+\varepsilon_{n+1})\sqrt{a} = \left(1+\frac{\varepsilon_n^2}{2(1+\varepsilon_n)}\right)\sqrt{a}
///   \quad\Rightarrow\quad
///   \varepsilon_n=\sqrt{\varepsilon_{n+1}^2+2\varepsilon_{n+1}}+\varepsilon_{n+1}
///   $$
///
///   - if $\varepsilon_0=1$,
///     - $\varepsilon_1=1/4=2^{-2}$
///     - $\varepsilon_2=1/40\simeq 2^{-5.321928094887363}$
///     - $\varepsilon_3=1/3280\simeq 2^{-11.679480099505446}$
///     - $\varepsilon_4=1/21523360\simeq 2^{-24.359399978023866}$
///     - $\varepsilon_5=1/926510094425920\simeq 2^{-49.718800023076994}$
///     - $\varepsilon_6=1/1716841910146256242328924544640\simeq 2^{-100.43760004615399}$
///   - if $\varepsilon_3=2^{-6}=0.015625$,
///     - $\varepsilon_2\simeq 2^{-2.372648026468475}\simeq 0.19309088580625855$
///     - $\varepsilon_1\simeq 2^{-0.24496984730140273}\simeq 0.8438334322821769$
///     - $\varepsilon_0\simeq 2^{1.2587823880924403}\simeq 2.392936955614776$
pub fn sqrt_floor<U>(a: U) -> U
where
    U: Copy
        + std::convert::From<u8>
        + std::convert::Into<u128>
        + std::ops::Add<Output = U>
        + std::ops::Sub<Output = U>
        + std::ops::Mul<Output = U>
        + std::ops::Div<Output = U>
        + std::ops::Shl<u32, Output = U>
        + std::ops::Shr<u32, Output = U>
        + std::ops::Not<Output = U>
        + std::cmp::Ord,
{
    if a <= U::from(1) {
        return a;
    }
    let bits = std::convert::Into::<u128>::into(!U::from(0)).count_ones();
    let mut k = 128 - std::convert::Into::<u128>::into(a - U::from(1)).leading_zeros();
    k = (k >> 1) + (k & 1);
    // s = 2^k
    let mut s: U = U::from(1) << k;
    // t = (s + a/s) / 2
    let mut t: U = (s + (a >> k)) >> 1;
    // s > floor(sqrt(x)) -> floor(sqrt(x)) <= t < s
    // s == floor(sqrt(x)) -> s == floor(sqrt(x)) <= t <= floor(sqrt(x)) + 1
    while s > t {
        if k < 4 {
            // fast exit:
            return if t >= (U::from(1) << (bits >> 1)) || t * t > a {
                t - U::from(1)
            } else {
                t
            };
        }
        s = t;
        t = (s + a / s) >> 1;
        k = (k + 1) >> 1;
    }
    s
}

/// Performs CRT (Chinese Remainder Theorem).
///
/// Given two sequences $r, m$ of length $n$, this function solves the modular equation system
///
/// \\[
///   x \equiv r_i \pmod{m_i}, \forall i \in \\{0, 1, \cdots, n - 1\\}
/// \\]
///
/// If there is no solution, it returns $(0, 0)$.
///
/// Otherwise, all of the solutions can be written as the form $x \equiv y \pmod z$, using integer $y, z\\ (0 \leq y < z = \text{lcm}(m))$.
/// It returns this $(y, z)$.
///
/// If $n = 0$, it returns $(0, 1)$.
///
/// # Constraints
///
/// - $|r| = |m|$
/// - $1 \leq m_{\forall i}$
/// - $\text{lcm}(m)$ is in `i64`
///
/// # Panics
///
/// Panics if the above constraints are not satisfied.
///
/// # Complexity
///
/// - $O(n \log \text{lcm}(m))$
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// let r = [2, 3, 2];
/// let m = [3, 5, 7];
/// assert_eq!(math::crt(&r, &m), (23, 105));
/// ```
pub fn crt(r: &[u64], m: &[u64]) -> (u64, u64) {
    assert_eq!(r.len(), m.len());
    // Contracts: 0 <= r0 < m0
    let (mut r0, mut m0) = (0, 1);
    for (&(mut ri), &(mut mi)) in r.iter().zip(m.iter()) {
        assert!(1 <= mi);
        ri %= mi;
        if m0 < mi {
            swap(&mut r0, &mut ri);
            swap(&mut m0, &mut mi);
        }
        if m0 % mi == 0 {
            if r0 % mi != ri {
                return (0, 0);
            }
            continue;
        }
        // assume: m0 > mi, lcm(m0, mi) >= 2 * max(m0, mi)

        // (r0, m0), (ri, mi) -> (r2, m2 = lcm(m0, m1));
        // r2 % m0 = r0
        // r2 % mi = ri
        // -> (r0 + x*m0) % mi = ri
        // -> x*u0*g = ri-r0 (mod u1*g) (u0*g = m0, u1*g = mi)
        // -> x = (ri - r0) / g * inv(u0) (mod u1)

        // im = inv(u0) (mod u1) (0 <= im < u1)
        let (g, im) = internal_math::inv_gcd(m0, mi);
        let u1 = mi / g;
        // ml = lcm(m0, mi)
        let ml = m0 * u1;
        // rd = ri - r0 (mod lcm(m0, mi))
        let rd = match ri.overflowing_sub(r0) {
            (v, true) => v.wrapping_add(ml),
            (v, false) => v,
        };
        // |ri - r0| < (m0 + mi) <= lcm(m0, mi)
        if rd % g != 0 {
            return (0, 0);
        }
        // u1 * u1 <= mi * mi / g / g <= m0 * mi / g = lcm(m0, mi)
        let x = rd / g % u1 * im % u1;

        // |r0| + |m0 * x|
        // < m0 + m0 * (u1 - 1)
        // = m0 + m0 * mi / g - m0
        // = lcm(m0, mi)
        r0 += x * m0;
        m0 = ml; // -> lcm(m0, mi)
    }

    (r0, m0)
}

/// Returns $\sum_{i = 0}^{n - 1} \lfloor \frac{a \times i + b}{m} \rfloor$.
///
/// # Constraints
///
/// - $0 \leq n \leq 10^9$
/// - $1 \leq m \leq 10^9$
/// - $0 \leq a, b \leq m$
///
/// # Panics
///
/// Panics if the above constraints are not satisfied and overflow or division by zero occurred.
///
/// # Complexity
///
/// - $O(\log(n + m + a + b))$
///
/// # Example
///
/// ```
/// use ac_library_rs::math;
///
/// assert_eq!(math::floor_sum(6, 5, 4, 3), 13);
/// ```
pub fn floor_sum(n: u64, m: u64, mut a: u64, mut b: u64) -> u64 {
    let mut ans = 0;
    if a >= m {
        ans += (n - 1) * n / 2 * (a / m);
        a %= m;
    }
    if b >= m {
        ans += n * (b / m);
        b %= m;
    }

    let y_max = a * n + b;
    if y_max < m {
        return ans;
    }
    ans += floor_sum(y_max / m, a, m, y_max % m);
    ans
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal)]
    #![allow(clippy::cognitive_complexity)]
    use super::*;
    #[test]
    fn test_pow_mod() {
        assert_eq!(pow_mod(0, 0, 1), 0);
        assert_eq!(pow_mod(0, 0, 3), 1);
        assert_eq!(pow_mod(0, 0, 723), 1);
        assert_eq!(pow_mod(0, 0, 998244353), 1);
        assert_eq!(pow_mod(0, 0, 2u32.pow(31)), 1);

        assert_eq!(pow_mod(0, 1, 1), 0);
        assert_eq!(pow_mod(0, 1, 3), 0);
        assert_eq!(pow_mod(0, 1, 723), 0);
        assert_eq!(pow_mod(0, 1, 998244353), 0);
        assert_eq!(pow_mod(0, 1, 2u32.pow(31)), 0);

        assert_eq!(pow_mod(0, i64::max_value() as u64, 1), 0);
        assert_eq!(pow_mod(0, i64::max_value() as u64, 3), 0);
        assert_eq!(pow_mod(0, i64::max_value() as u64, 723), 0);
        assert_eq!(pow_mod(0, i64::max_value() as u64, 998244353), 0);
        assert_eq!(pow_mod(0, i64::max_value() as u64, 2u32.pow(31)), 0);

        assert_eq!(pow_mod(1, 0, 1), 0);
        assert_eq!(pow_mod(1, 0, 3), 1);
        assert_eq!(pow_mod(1, 0, 723), 1);
        assert_eq!(pow_mod(1, 0, 998244353), 1);
        assert_eq!(pow_mod(1, 0, 2u32.pow(31)), 1);

        assert_eq!(pow_mod(1, 1, 1), 0);
        assert_eq!(pow_mod(1, 1, 3), 1);
        assert_eq!(pow_mod(1, 1, 723), 1);
        assert_eq!(pow_mod(1, 1, 998244353), 1);
        assert_eq!(pow_mod(1, 1, 2u32.pow(31)), 1);

        assert_eq!(pow_mod(1, i64::max_value() as u64, 1), 0);
        assert_eq!(pow_mod(1, i64::max_value() as u64, 3), 1);
        assert_eq!(pow_mod(1, i64::max_value() as u64, 723), 1);
        assert_eq!(pow_mod(1, i64::max_value() as u64, 998244353), 1);
        assert_eq!(pow_mod(1, i64::max_value() as u64, 2u32.pow(31)), 1);

        assert_eq!(pow_mod(i64::max_value() as u64, 0, 1), 0);
        assert_eq!(pow_mod(i64::max_value() as u64, 0, 3), 1);
        assert_eq!(pow_mod(i64::max_value() as u64, 0, 723), 1);
        assert_eq!(pow_mod(i64::max_value() as u64, 0, 998244353), 1);
        assert_eq!(pow_mod(i64::max_value() as u64, 0, 2u32.pow(31)), 1);

        assert_eq!(
            pow_mod(i64::max_value() as u64, i64::max_value() as u64, 1),
            0
        );
        assert_eq!(
            pow_mod(i64::max_value() as u64, i64::max_value() as u64, 3),
            1
        );
        assert_eq!(
            pow_mod(i64::max_value() as u64, i64::max_value() as u64, 723),
            640
        );
        assert_eq!(
            pow_mod(i64::max_value() as u64, i64::max_value() as u64, 998244353),
            683296792
        );
        assert_eq!(
            pow_mod(
                i64::max_value() as u64,
                i64::max_value() as u64,
                2u32.pow(31)
            ),
            2147483647
        );

        assert_eq!(pow_mod(2, 3, 1_000_000_007), 8);
        assert_eq!(pow_mod(5, 7, 1_000_000_007), 78125);
        assert_eq!(pow_mod(123, 456, 1_000_000_007), 565291922);
    }

    #[test]
    fn test_ipow_mod() {
        assert_eq!(ipow_mod(0, 0, 1), 0);
        assert_eq!(ipow_mod(0, 0, 3), 1);
        assert_eq!(ipow_mod(0, 0, 723), 1);
        assert_eq!(ipow_mod(0, 0, 998244353), 1);
        assert_eq!(ipow_mod(0, 0, 2u32.pow(31)), 1);

        assert_eq!(ipow_mod(0, 1, 1), 0);
        assert_eq!(ipow_mod(0, 1, 3), 0);
        assert_eq!(ipow_mod(0, 1, 723), 0);
        assert_eq!(ipow_mod(0, 1, 998244353), 0);
        assert_eq!(ipow_mod(0, 1, 2u32.pow(31)), 0);

        assert_eq!(ipow_mod(0, i64::max_value() as u64, 1), 0);
        assert_eq!(ipow_mod(0, i64::max_value() as u64, 3), 0);
        assert_eq!(ipow_mod(0, i64::max_value() as u64, 723), 0);
        assert_eq!(ipow_mod(0, i64::max_value() as u64, 998244353), 0);
        assert_eq!(ipow_mod(0, i64::max_value() as u64, 2u32.pow(31)), 0);

        assert_eq!(ipow_mod(1, 0, 1), 0);
        assert_eq!(ipow_mod(1, 0, 3), 1);
        assert_eq!(ipow_mod(1, 0, 723), 1);
        assert_eq!(ipow_mod(1, 0, 998244353), 1);
        assert_eq!(ipow_mod(1, 0, 2u32.pow(31)), 1);

        assert_eq!(ipow_mod(1, 1, 1), 0);
        assert_eq!(ipow_mod(1, 1, 3), 1);
        assert_eq!(ipow_mod(1, 1, 723), 1);
        assert_eq!(ipow_mod(1, 1, 998244353), 1);
        assert_eq!(ipow_mod(1, 1, 2u32.pow(31)), 1);

        assert_eq!(ipow_mod(1, i64::max_value() as u64, 1), 0);
        assert_eq!(ipow_mod(1, i64::max_value() as u64, 3), 1);
        assert_eq!(ipow_mod(1, i64::max_value() as u64, 723), 1);
        assert_eq!(ipow_mod(1, i64::max_value() as u64, 998244353), 1);
        assert_eq!(ipow_mod(1, i64::max_value() as u64, 2u32.pow(31)), 1);

        assert_eq!(ipow_mod(i64::max_value(), 0, 1), 0);
        assert_eq!(ipow_mod(i64::max_value(), 0, 3), 1);
        assert_eq!(ipow_mod(i64::max_value(), 0, 723), 1);
        assert_eq!(ipow_mod(i64::max_value(), 0, 998244353), 1);
        assert_eq!(ipow_mod(i64::max_value(), 0, 2u32.pow(31)), 1);

        assert_eq!(ipow_mod(i64::max_value(), i64::max_value() as u64, 1), 0);
        assert_eq!(ipow_mod(i64::max_value(), i64::max_value() as u64, 3), 1);
        assert_eq!(
            ipow_mod(i64::max_value(), i64::max_value() as u64, 723),
            640
        );
        assert_eq!(
            ipow_mod(i64::max_value(), i64::max_value() as u64, 998244353),
            683296792
        );
        assert_eq!(
            ipow_mod(i64::max_value(), i64::max_value() as u64, 2u32.pow(31)),
            2147483647
        );

        assert_eq!(ipow_mod(2, 3, 1_000_000_007), 8);
        assert_eq!(ipow_mod(5, 7, 1_000_000_007), 78125);
        assert_eq!(ipow_mod(123, 456, 1_000_000_007), 565291922);
    }

    #[test]
    #[should_panic]
    fn test_inv_mod_1() {
        inv_mod::<u64>(271828, 0);
    }

    #[test]
    #[should_panic]
    fn test_inv_mod_2() {
        inv_mod::<u64>(3141592, 1000000008);
    }

    #[test]
    fn test_crt() {
        let a = [44, 23, 13];
        let b = [13, 50, 22];
        assert_eq!(crt(&a, &b), (1773, 7150));
        let a = [12345, 67890, 99999];
        let b = [13, 444321, 95318];
        assert_eq!(crt(&a, &b), (103333581255, 550573258014));
        let a = [0, 3, 4];
        let b = [1, 9, 5];
        assert_eq!(crt(&a, &b), (39, 45));
    }

    #[test]
    fn test_floor_sum() {
        assert_eq!(floor_sum(0, 1, 0, 0), 0);
        assert_eq!(floor_sum(1_000_000_000, 1, 1, 1), 500_000_000_500_000_000);
        assert_eq!(
            floor_sum(1_000_000_000, 1_000_000_000, 999_999_999, 999_999_999),
            499_999_999_500_000_000
        );
        assert_eq!(floor_sum(332955, 5590132, 2231, 999423), 22014575);
    }

    #[test]
    fn test_sqrt_floor_u64() {
        let sqrt = sqrt_floor::<u64>;
        assert_eq!(sqrt(0), 0);
        assert_eq!(sqrt(1), 1);
        assert_eq!(sqrt(2), 1);
        assert_eq!(sqrt(3), 1);
        assert_eq!(sqrt(4), 2);
        assert_eq!(sqrt(5), 2);
        assert_eq!(sqrt(6), 2);
        assert_eq!(sqrt(7), 2);
        assert_eq!(sqrt(8), 2);
        assert_eq!(sqrt(9), 3);
        assert_eq!(sqrt(10), 3);
        assert_eq!(sqrt(65), 8);
        assert_eq!(sqrt(257), 16);
        assert_eq!(sqrt(65_537), 256);
        assert_eq!(sqrt(4_294_967_297), 65_536);
        assert_eq!(sqrt(999_999_999_999_999_999), 999_999_999);
        assert_eq!(sqrt(1_000_000_000_000_000_000), 1_000_000_000);
        assert_eq!(sqrt(1_000_000_000_000_000_001), 1_000_000_000);
        assert_eq!(sqrt(4_611_686_018_427_387_903), 2_147_483_647);
        assert_eq!(sqrt(4_611_686_018_427_387_904), 2_147_483_648);
        assert_eq!(sqrt(4_611_686_018_427_387_905), 2_147_483_648);
        assert_eq!(sqrt(4_611_686_018_427_387_906), 2_147_483_648);
        assert_eq!(sqrt(10_000_000_000_000_000_000), 3_162_277_660);
        assert_eq!(sqrt(18_446_744_065_119_617_024), 4_294_967_294);
        assert_eq!(sqrt(18_446_744_065_119_617_025), 4_294_967_295);
        assert_eq!(sqrt(18_446_744_065_119_617_026), 4_294_967_295);
        assert_eq!(sqrt(18_446_744_065_119_617_027), 4_294_967_295);
        assert_eq!(sqrt(u64::max_value()), 4_294_967_295);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sqrt_floor_u64_rand() {
        use crate::internal_math::x86_rdrand;
        for _ in 0..10000 {
            let n = x86_rdrand(u64::max_value());
            let sqrtn = sqrt_floor(n);
            assert!(sqrtn * sqrtn <= n);
            assert!(n < (sqrtn + 1).saturating_mul(sqrtn + 1));
        }
    }

    #[test]
    fn test_sqrt_floor_u64_seq() {
        for n in 1..429496 {
            assert_eq!(sqrt_floor::<u64>(n * n + 1), n, "{}", n);
        }
    }
}
