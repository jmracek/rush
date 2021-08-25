# Rush

Rush will be an RPC service built around a locality sensitive hash database.  Locality sensitive hashing is an algorithm for finding approximate nearest neighbours.  Applications of LSH are mostly centered around information retrieval; for example, given an embedding representation of some media (images, audio, or text), locality sensitive hashing may be used to retrieve the most similar documents in a database to the requested one.

## Installation

At the moment Rush builds only for x86_64 targets.  You can build Rush as follows:
```
git clone git@github.com:jmracek/rush.git && cd rush && RUSTFLAGS="-C target-feature=+sse2,+avx2,+fma -C overflow-checks=off" cargo build --release
```

## Example Use

Currently the binary produced after building Rush doesn't do anything.  That will change soon-ish.  Eventually, you will be able to use Rush as follows:
```
./rush --dataset [wildcarded-path-to-data].proto.bin --port 8008
```
I will also set up a build using Docker to enable deployment on Kubernetes.  The configurations and Dockerfiles for this don't exist yet, but will by the time I am ready to share this project with the world.

## Contributing

If you like this project and you're reading this, you're likely a better Rust programmer than I am.  If you have major features you would like to include, please open an issue.  Since I'm not expecting anyone to actually read this and open an issue, please also e-mail me at james.a.mracek@gmail.com to give me a heads up and we'll discuss your ideas.  For minor issues and fixes, please feel free to open a PR and send me an e-mail.


## Support and Bugs

E-mail me at james.a.mracek@gmail.com or open a Github issue.

## License

MIT


