// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>

#include "ACTFW/GenericDetector/GenericDetectorElement.hpp"
#include "ACTFW/GenericDetector/ProtoLayerCreatorT.hpp"
#include "Acts/Detector/DetectorElementBase.hpp"
#include "Acts/Layers/ApproachDescriptor.hpp"
#include "Acts/Layers/Layer.hpp"
#include "Acts/Layers/ProtoLayer.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Material/MaterialProperties.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Tools/ILayerBuilder.hpp"
#include "Acts/Tools/LayerCreator.hpp"
#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Utilities/GeometryContext.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Logger.hpp"

namespace FW {
namespace Generic {

  using Acts::VectorHelpers::eta;
  using Acts::VectorHelpers::phi;
  using Acts::VectorHelpers::perp;

  typedef std::pair<const Acts::Surface*, Acts::Vector3D> SurfacePosition;

  /// @class LayerBuilderT
  ///
  /// The LayerBuilderT is able to build cylinder & disc layers
  /// from hard-coded input.
  /// This is ment for the simple detector examples.
  template <typename detector_element_t>
  class LayerBuilderT : public Acts::ILayerBuilder
  {
  public:
    /// @struct Config
    /// Nested configuration struct for the LayerBuilderT
    struct Config
    {
      /// The string based identification
      std::string layerIdentification = "";

      /// The pre-produced proto layers for the central part
      std::vector<ProtoLayerSurfaces> centralProtoLayers;

      /// the material concentration: -1 inner, 0 central, 1 outer
      std::vector<int> centralLayerMaterialConcentration;
      /// the assigned material propertis @todo change to surface material
      std::vector<Acts::MaterialProperties> centralLayerMaterialProperties;

      /// The pre-produced proto layers for the negative part
      std::vector<ProtoLayerSurfaces> negativeProtoLayers;

      /// The pre-produced proto layers for the positive part
      std::vector<ProtoLayerSurfaces> positiveProtoLayers;

      /// the material concentration: -1 inner, 0 central, 1 outer
      std::vector<int> posnegLayerMaterialConcentration;

      /// the material prooperties @todo change to surface material
      std::vector<Acts::MaterialProperties> posnegLayerMaterialProperties;

      /// helper tools: layer creator
      std::shared_ptr<const Acts::LayerCreator> layerCreator = nullptr;
      /// helper tools: central passive layer builder
      std::shared_ptr<const Acts::ILayerBuilder> centralPassiveLayerBuilder
          = nullptr;
      /// helper tools: p/n passive layer builder
      std::shared_ptr<const Acts::ILayerBuilder> posnegPassiveLayerBuilder
          = nullptr;
    };

    /// Constructor
    /// @param glbConfig is the configuration class
    LayerBuilderT(const Config&                       cfg,
                  std::unique_ptr<const Acts::Logger> logger
                  = Acts::getDefaultLogger("LayerBuilderT",
                                           Acts::Logging::INFO));

    /// LayerBuilder interface method - returning the layers at negative side
    const Acts::LayerVector
    negativeLayers(const Acts::GeometryContext& gctx) const final override;

    /// LayerBuilder interface method - returning the central layers
    const Acts::LayerVector
    centralLayers(const Acts::GeometryContext& gctx) const final override;

    /// LayerBuilder interface method - returning the layers at positive side
    const Acts::LayerVector
    positiveLayers(const Acts::GeometryContext& gctx) const final override;

    /// ILayerBuilder method
    const std::string&
    identification() const final override
    {
      return m_cfg.layerIdentification;
    }

  private:
    const Acts::LayerVector
    constructEndcapLayers(const Acts::GeometryContext& gctx, int side) const;

    /// Configuration member
    Config m_cfg;

    /// Private access to the logging instance
    const Acts::Logger&
    logger() const
    {
      return *m_logger;
    }

    /// the logging instance
    std::unique_ptr<const Acts::Logger> m_logger;
  };

  template <typename detector_element_t>
  const Acts::LayerVector
  LayerBuilderT<detector_element_t>::centralLayers(
      const Acts::GeometryContext& gctx) const
  {
    // create the vector
    Acts::LayerVector cLayers;
    cLayers.reserve(m_cfg.centralProtoLayers.size());
    // the layer counter
    size_t icl = 0;
    for (auto& cpl : m_cfg.centralProtoLayers) {
      // create the layer actually
      Acts::MutableLayerPtr cLayer = m_cfg.layerCreator->cylinderLayer(
          gctx, cpl.surfaces, cpl.bins0, cpl.bins1, cpl.protoLayer);

      // the layer is built let's see if it needs material
      if (m_cfg.centralLayerMaterialProperties.size()) {
        // get the material from configuration
        Acts::MaterialProperties layerMaterialProperties
            = m_cfg.centralLayerMaterialProperties.at(icl);
        std::shared_ptr<const Acts::SurfaceMaterial> layerMaterialPtr(
            new Acts::HomogeneousSurfaceMaterial(layerMaterialProperties));
        // central material
        if (m_cfg.centralLayerMaterialConcentration.at(icl) == 0.) {
          // the layer surface is the material surface
          cLayer->surfaceRepresentation().setAssociatedMaterial(
              layerMaterialPtr);
          ACTS_VERBOSE("- and material at central layer surface.");
        } else {
          // approach surface material
          // get the approach descriptor - at this stage we know that the
          // approachDescriptor exists
          auto approachSurfaces
              = cLayer->approachDescriptor()->containedSurfaces();
          if (m_cfg.centralLayerMaterialConcentration.at(icl) > 0) {
            auto mutableOuterSurface
                = const_cast<Acts::Surface*>(approachSurfaces.at(1));
            mutableOuterSurface->setAssociatedMaterial(layerMaterialPtr);
            ACTS_VERBOSE("- and material at outer approach surface");
          } else {
            auto mutableInnerSurface
                = const_cast<Acts::Surface*>(approachSurfaces.at(0));
            mutableInnerSurface->setAssociatedMaterial(layerMaterialPtr);
            ACTS_VERBOSE("- and material at inner approach surface");
          }
        }
      }
      // push it into the layer vector
      cLayers.push_back(cLayer);
      ++icl;
    }
    return cLayers;
  }

  template <typename detector_element_t>
  const Acts::LayerVector
  LayerBuilderT<detector_element_t>::negativeLayers(
      const Acts::GeometryContext& gctx) const
  {
    return constructEndcapLayers(gctx, -1);
  }

  template <typename detector_element_t>
  const Acts::LayerVector
  LayerBuilderT<detector_element_t>::positiveLayers(
      const Acts::GeometryContext& gctx) const
  {
    return constructEndcapLayers(gctx, 1);
  }

  template <typename detector_element_t>
  LayerBuilderT<detector_element_t>::LayerBuilderT(
      const LayerBuilderT<detector_element_t>::Config& cfg,
      std::unique_ptr<const Acts::Logger>              log)
    : Acts::ILayerBuilder(), m_cfg(cfg), m_logger(std::move(log))
  {
  }

  template <typename detector_element_t>
  const Acts::LayerVector
  LayerBuilderT<detector_element_t>::constructEndcapLayers(
      const Acts::GeometryContext& gctx,
      int                          side) const
  {

    // The from negative or positive proto layers
    const auto& protoLayers
        = (side < 0) ? m_cfg.negativeProtoLayers : m_cfg.positiveProtoLayers;

    // create the vector
    Acts::LayerVector eLayers;
    eLayers.reserve(protoLayers.size());

    // the layer counter
    size_t ipnl = 0;
    // loop over the proto layers and create the actual layers
    for (auto& ple : protoLayers) {
      /// the layer is created
      Acts::MutableLayerPtr eLayer = m_cfg.layerCreator->discLayer(
          gctx, ple.surfaces, ple.bins0, ple.bins1, ple.protoLayer);

      // the layer is built let's see if it needs material
      if (m_cfg.posnegLayerMaterialProperties.size()) {
        std::shared_ptr<const Acts::SurfaceMaterial> layerMaterialPtr(
            new Acts::HomogeneousSurfaceMaterial(
                m_cfg.posnegLayerMaterialProperties[ipnl]));
        // central material
        if (m_cfg.posnegLayerMaterialConcentration.at(ipnl) == 0.) {
          // assign the surface material - the layer surface is the material
          // surface
          eLayer->surfaceRepresentation().setAssociatedMaterial(
              layerMaterialPtr);
          ACTS_VERBOSE("- and material at central layer surface.");
        } else {
          // approach surface material
          // get the approach descriptor - at this stage we know that the
          // approachDescriptor exists
          auto approachSurfaces
              = eLayer->approachDescriptor()->containedSurfaces();
          if (m_cfg.posnegLayerMaterialConcentration.at(ipnl) > 0.) {
            int  sf = side < 0 ? 0 : 1;
            auto mutableInnerSurface
                = const_cast<Acts::Surface*>(approachSurfaces.at(sf));
            mutableInnerSurface->setAssociatedMaterial(layerMaterialPtr);
            ACTS_VERBOSE("- and material at outer approach surfaces.");
          } else {
            int  sf = side < 0 ? 1 : 0;
            auto mutableOuterSurface
                = const_cast<Acts::Surface*>(approachSurfaces.at(sf));
            mutableOuterSurface->setAssociatedMaterial(layerMaterialPtr);
            ACTS_VERBOSE("- and material at inner approach surfaces.");
          }
        }
      }
      // push it into the layer vector
      eLayers.push_back(eLayer);
      ++ipnl;
    }
    return eLayers;
  }

}  // end of namespace Generic
}  // end of namespace FW
