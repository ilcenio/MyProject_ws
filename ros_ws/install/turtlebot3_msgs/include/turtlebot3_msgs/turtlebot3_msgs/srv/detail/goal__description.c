// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#include "turtlebot3_msgs/srv/detail/goal__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
const rosidl_type_hash_t *
turtlebot3_msgs__srv__Goal__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x03, 0x14, 0xcb, 0x41, 0x23, 0xe9, 0x95, 0xfe,
      0x49, 0xf3, 0x34, 0x5f, 0x6c, 0xf2, 0x37, 0x02,
      0x37, 0x9d, 0xbb, 0x07, 0x67, 0x62, 0xf0, 0x59,
      0x50, 0xa0, 0x46, 0xf2, 0x52, 0x3e, 0x5c, 0xbf,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
const rosidl_type_hash_t *
turtlebot3_msgs__srv__Goal_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xf1, 0x7d, 0x1a, 0x93, 0x07, 0x7e, 0x3b, 0x75,
      0x47, 0xf2, 0x93, 0x86, 0x6d, 0x76, 0x1b, 0xf4,
      0xb1, 0x95, 0xa4, 0x27, 0xc1, 0xcf, 0xe2, 0x54,
      0x0b, 0x00, 0xf4, 0x19, 0xed, 0xb9, 0xf3, 0xce,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
const rosidl_type_hash_t *
turtlebot3_msgs__srv__Goal_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x85, 0x94, 0x4f, 0x71, 0x18, 0x4b, 0x8e, 0xd2,
      0x86, 0xd8, 0xe4, 0xf8, 0xa7, 0xc7, 0xd0, 0xb2,
      0x28, 0xc4, 0x90, 0xf9, 0xf0, 0xc0, 0xa1, 0xb6,
      0x5d, 0x11, 0x51, 0xb7, 0x77, 0xac, 0x1e, 0x6d,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_turtlebot3_msgs
const rosidl_type_hash_t *
turtlebot3_msgs__srv__Goal_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x3d, 0xf7, 0xee, 0x34, 0xdc, 0x01, 0xed, 0x3f,
      0xb8, 0x00, 0x27, 0x1f, 0x56, 0x65, 0x5f, 0x75,
      0x40, 0x31, 0xd9, 0x2b, 0xdc, 0x08, 0x45, 0x91,
      0x45, 0xde, 0x16, 0x3d, 0x8e, 0xe2, 0x6a, 0x23,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char turtlebot3_msgs__srv__Goal__TYPE_NAME[] = "turtlebot3_msgs/srv/Goal";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char turtlebot3_msgs__srv__Goal_Event__TYPE_NAME[] = "turtlebot3_msgs/srv/Goal_Event";
static char turtlebot3_msgs__srv__Goal_Request__TYPE_NAME[] = "turtlebot3_msgs/srv/Goal_Request";
static char turtlebot3_msgs__srv__Goal_Response__TYPE_NAME[] = "turtlebot3_msgs/srv/Goal_Response";

// Define type names, field names, and default values
static char turtlebot3_msgs__srv__Goal__FIELD_NAME__request_message[] = "request_message";
static char turtlebot3_msgs__srv__Goal__FIELD_NAME__response_message[] = "response_message";
static char turtlebot3_msgs__srv__Goal__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field turtlebot3_msgs__srv__Goal__FIELDS[] = {
  {
    {turtlebot3_msgs__srv__Goal__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {turtlebot3_msgs__srv__Goal_Event__TYPE_NAME, 30, 30},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription turtlebot3_msgs__srv__Goal__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Event__TYPE_NAME, 30, 30},
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
turtlebot3_msgs__srv__Goal__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {turtlebot3_msgs__srv__Goal__TYPE_NAME, 24, 24},
      {turtlebot3_msgs__srv__Goal__FIELDS, 3, 3},
    },
    {turtlebot3_msgs__srv__Goal__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = turtlebot3_msgs__srv__Goal_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = turtlebot3_msgs__srv__Goal_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = turtlebot3_msgs__srv__Goal_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char turtlebot3_msgs__srv__Goal_Request__FIELD_NAME__structure_needs_at_least_one_member[] = "structure_needs_at_least_one_member";

static rosidl_runtime_c__type_description__Field turtlebot3_msgs__srv__Goal_Request__FIELDS[] = {
  {
    {turtlebot3_msgs__srv__Goal_Request__FIELD_NAME__structure_needs_at_least_one_member, 35, 35},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
turtlebot3_msgs__srv__Goal_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
      {turtlebot3_msgs__srv__Goal_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__pose_x[] = "pose_x";
static char turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__pose_y[] = "pose_y";
static char turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__success[] = "success";

static rosidl_runtime_c__type_description__Field turtlebot3_msgs__srv__Goal_Response__FIELDS[] = {
  {
    {turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__pose_x, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__pose_y, 6, 6},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_FLOAT,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
turtlebot3_msgs__srv__Goal_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
      {turtlebot3_msgs__srv__Goal_Response__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__info[] = "info";
static char turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__request[] = "request";
static char turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field turtlebot3_msgs__srv__Goal_Event__FIELDS[] = {
  {
    {turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
    },
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription turtlebot3_msgs__srv__Goal_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
    {NULL, 0, 0},
  },
  {
    {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
turtlebot3_msgs__srv__Goal_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {turtlebot3_msgs__srv__Goal_Event__TYPE_NAME, 30, 30},
      {turtlebot3_msgs__srv__Goal_Event__FIELDS, 3, 3},
    },
    {turtlebot3_msgs__srv__Goal_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = turtlebot3_msgs__srv__Goal_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = turtlebot3_msgs__srv__Goal_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "---\n"
  "float32 pose_x\n"
  "float32 pose_y\n"
  "bool success";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
turtlebot3_msgs__srv__Goal__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {turtlebot3_msgs__srv__Goal__TYPE_NAME, 24, 24},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 47, 47},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
turtlebot3_msgs__srv__Goal_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {turtlebot3_msgs__srv__Goal_Request__TYPE_NAME, 32, 32},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
turtlebot3_msgs__srv__Goal_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {turtlebot3_msgs__srv__Goal_Response__TYPE_NAME, 33, 33},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
turtlebot3_msgs__srv__Goal_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {turtlebot3_msgs__srv__Goal_Event__TYPE_NAME, 30, 30},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
turtlebot3_msgs__srv__Goal__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *turtlebot3_msgs__srv__Goal__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *turtlebot3_msgs__srv__Goal_Event__get_individual_type_description_source(NULL);
    sources[4] = *turtlebot3_msgs__srv__Goal_Request__get_individual_type_description_source(NULL);
    sources[5] = *turtlebot3_msgs__srv__Goal_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
turtlebot3_msgs__srv__Goal_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *turtlebot3_msgs__srv__Goal_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
turtlebot3_msgs__srv__Goal_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *turtlebot3_msgs__srv__Goal_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
turtlebot3_msgs__srv__Goal_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *turtlebot3_msgs__srv__Goal_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *turtlebot3_msgs__srv__Goal_Request__get_individual_type_description_source(NULL);
    sources[4] = *turtlebot3_msgs__srv__Goal_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
