// Check Problem Statement via https://judge.yosupo.jp/problem/unionfind
use ac_library_rs::dsu::Dsu;
use std::io::prelude::*;

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let n: usize = input.next().unwrap().parse().unwrap();
    let q: usize = input.next().unwrap().parse().unwrap();

    let mut dsu = Dsu::new(n);

    for _ in 0..q {
        let kind = input.next().unwrap().parse().unwrap();
        let u: usize = input.next().unwrap().parse().unwrap();
        let v: usize = input.next().unwrap().parse().unwrap();
        match kind {
            0 => {
                dsu.merge(u, v);
            }
            1 => {
                println!("{}", u8::from(dsu.same(u, v)));
            }
            _ => {}
        }
    }
}
