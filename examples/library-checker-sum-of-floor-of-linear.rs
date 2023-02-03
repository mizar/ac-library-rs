// Check Problem Statement via https://judge.yosupo.jp/problem/sum_of_floor_of_linear
use ac_library_rs::math;
use std::io::prelude::*;

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let t: usize = input.next().unwrap().parse().unwrap();

    for _ in 0..t {
        let n: u64 = input.next().unwrap().parse().unwrap();
        let m: u64 = input.next().unwrap().parse().unwrap();
        let a: u64 = input.next().unwrap().parse().unwrap();
        let b: u64 = input.next().unwrap().parse().unwrap();
        println!("{}", math::floor_sum(n, m, a, b));
    }
}
