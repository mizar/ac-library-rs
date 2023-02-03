// Check Problem Statement via https://judge.yosupo.jp/problem/convolution_mod
use ac_library_rs::{convolution, modint::ModInt998244353 as Mint};
use std::io::prelude::*;

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let n = input.next().unwrap().parse().unwrap();
    let m = input.next().unwrap().parse().unwrap();
    let a: Vec<Mint> = input
        .by_ref()
        .take(n)
        .map(str::parse)
        .map(Result::unwrap)
        .collect();
    let b: Vec<Mint> = input
        .by_ref()
        .take(m)
        .map(str::parse)
        .map(Result::unwrap)
        .collect();

    print_oneline(convolution::convolution(&a, &b));
}

fn print_oneline<I: IntoIterator<Item = T>, T: std::fmt::Display>(values: I) {
    use std::fmt::Write;
    println!(
        "{}",
        values
            .into_iter()
            .fold(String::new(), |mut acc, cur| {
                write!(&mut acc, "{}", cur).unwrap();
                acc.push(' ');
                acc
            })
            .trim_end()
    );
}
