// This file is part of the Acts project.
//
// Copyright (C) 2017-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/program_options.hpp>
#include "ACTFW/DD4hepDetector/DD4hepDetector.hpp"
#include "detail/FatrasExampleBase.hpp"

/// @brief main executable
///
/// @param argc The argument count
/// @param argv The argument list
int
main(int argc, char* argv[])
{
  // ----------------------------------------------------------
  DD4hepOptions  dd4HepOptions;
  DD4hepGeometry dd4HepGeometry;

  // now process it
  return fatrasExample(argc, argv, dd4HepOptions, dd4HepGeometry);
}
