use std::iter::{repeat_with, FromIterator};
use std::ops::AddAssign;

// Reference: https://en.wikipedia.org/wiki/Fenwick_tree
pub struct FenwickTree<T> {
    n: usize,
    ary: Vec<T>,
}

impl<T: for<'a> std::ops::AddAssign<&'a T> + Default> FenwickTree<T> {
    pub fn new(n: usize) -> Self {
        FenwickTree {
            n,
            ary: repeat_with(T::default).take(n).collect(),
        }
    }
    pub fn accum(&self, mut idx: usize) -> T {
        let mut sum = T::default();
        while idx > 0 {
            sum += &self.ary[idx - 1];
            idx &= idx - 1;
        }
        sum
    }
    /// performs data[idx] += val;
    pub fn add<U>(&mut self, mut idx: usize, val: U)
    where
        T: for<'a> std::ops::AddAssign<&'a U>,
    {
        let n = self.n;
        idx += 1;
        while idx <= n {
            self.ary[idx - 1] += &val;
            idx += idx & idx.wrapping_neg();
        }
    }
    /// Returns data[l] + ... + data[r - 1].
    pub fn sum(&self, l: usize, r: usize) -> T
    where
        T: std::ops::Sub<Output = T>,
    {
        self.accum(r) - self.accum(l)
    }
}
impl<T: Clone + AddAssign<T>> From<Vec<T>> for FenwickTree<T> {
    fn from(mut ary: Vec<T>) -> Self {
        for i in 1..=ary.len() {
            let j = i + (i & i.wrapping_neg());
            if j <= ary.len() {
                let add = ary[i - 1].clone();
                ary[j - 1] += add;
            }
        }
        Self { n: ary.len(), ary }
    }
}
impl<T: Clone + AddAssign<T>> FromIterator<T> for FenwickTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenwick_tree_works() {
        let mut bit = FenwickTree::<i64>::new(5);
        // [1, 2, 3, 4, 5]
        for i in 0..5 {
            bit.add(i, i as i64 + 1);
        }
        assert_eq!(bit.sum(0, 5), 15);
        assert_eq!(bit.sum(0, 4), 10);
        assert_eq!(bit.sum(1, 3), 5);
    }

    #[test]
    fn from_iter_works() {
        let tree = FenwickTree::from_iter(vec![1, 2, 3, 4, 5].iter().map(|x| x * 2));
        let internal = vec![2, 4, 6, 8, 10];
        for j in 0..=internal.len() {
            for i in 0..=j {
                assert_eq!(tree.sum(i, j), internal[i..j].iter().sum::<i32>());
            }
        }
    }
}
