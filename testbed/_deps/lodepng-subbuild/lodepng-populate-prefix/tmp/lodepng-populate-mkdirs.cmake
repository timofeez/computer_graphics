# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-src"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-build"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/tmp"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/src"
  "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/timofey/Downloads/veekay/testbed/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
