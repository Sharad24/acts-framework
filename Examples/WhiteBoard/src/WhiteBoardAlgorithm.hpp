// This file is part of the Acts project.
//
// Copyright (C) 2017 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <memory>
#include "ACTFW/Framework/BareAlgorithm.hpp"
#include "ACTFW/Framework/ProcessCode.hpp"

namespace FW {

/// @class Algorithm
///
/// Test algorithm for the WhiteBoard writing/reading
///
class WhiteBoardAlgorithm : public FW::BareAlgorithm
{
public:
  /// @class Config
  /// Nested Configuration class for the WhiteBoardAlgorithm
  /// It extends the Algorithm::Config Class
  struct Config
  {
    // Input collection of DataClassOne (optional)
    std::string inputClassOneCollection = "";
    // Output collection of DataClassOne (optional)
    std::string outputClassOneCollection = "";
    // Input collection of DataClassTwo (optional)
    std::string inputClassTwoCollection = "";
    // Output collection of DataClassTwo (optional)
    std::string outputClassTwoCollection = "";
  };

  /// Constructor
  ///
  /// @param cfg is the configruation
  WhiteBoardAlgorithm(const Config&        cfg,
                      Acts::Logging::Level level = Acts::Logging::INFO);

  /// Framework execode method
  ///
  /// @param context is the AlgorithmContext context
  FW::ProcessCode
  execute(const AlgorithmContext& context) const final override;

private:
  Config m_cfg;
};

}  // namespace FW
