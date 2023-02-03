// Check Problem Statement via https://judge.yosupo.jp/problem/static_range_sum
use ac_library_rs::fenwicktree::FenwickTree;
use std::io::prelude::*;

#[allow(clippy::many_single_char_names)]
#[allow(clippy::needless_collect)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let n = input.next().unwrap().parse().unwrap();
    let q: usize = input.next().unwrap().parse().unwrap();

    let mut fenwick = FenwickTree::<u64>::new(n);
    for (i, a) in input
        .by_ref()
        .take(n)
        .map(str::parse)
        .map(Result::unwrap)
        .enumerate()
    {
        fenwick.add(i, a);
    }
    for _ in 0..q {
        let l = input.next().unwrap().parse().unwrap();
        let r = input.next().unwrap().parse().unwrap();
        println!("{}", fenwick.sum(l, r));
    }
}
